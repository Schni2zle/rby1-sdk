#!/usr/bin/env python3
"""
deploy_trajectory.py — Replay pickled whole-body trajectories on a physical RBY1.

Usage
-----
# Verify joint ordering first (no motion):
    python deploy_trajectory.py --address 192.168.1.1:50051 --print-joints

# Cautious first run at 25 % speed, trajectory index 0 only:
    python deploy_trajectory.py --address 192.168.1.1:50051 --traj-index 0 --speed-scale 0.25

# Full speed, all trajectories, arm + torso only (base & gripper disabled):
    python deploy_trajectory.py --address 192.168.1.1:50051

# Full speed, with gripper (requires DynamixelBus / UPC gripper wired):
    python deploy_trajectory.py --address 192.168.1.1:50051 --enable-gripper

# Move to zero pose before each trajectory and skip the confirmation prompt:
    python deploy_trajectory.py --address 192.168.1.1:50051 --go-home --no-confirm

CRITICAL BEFORE FIRST RUN
--------------------------
1. Run --print-joints and verify the torso / arm joint ordering printed matches
   TORSO_JOINT_NAMES and the expected 7-DOF left-arm layout.
2. Have a hardware e-stop reachable.
3. Use --speed-scale 0.25 for the first playback of any new trajectory set.
4. Confirm STEP_DT matches the physics step used during recording.
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import rby1_sdk as rby

# ──────────────────────────────────────────────────────────────────────────────
# Tuneable constants — verify against your hardware before running
# ──────────────────────────────────────────────────────────────────────────────

# Seconds per recorded physics step (20 Hz → 0.05 s).  Must match recording.
STEP_DT: float = 0.05

# Body command minimum_time = STEP_DT * this factor (gives controller headroom).
MOVE_TIME_FACTOR: float = 1.2

# control_hold_time for body joint-position commands.
BODY_HOLD_TIME: float = 1.0

# Extra hold time appended to base SE2 velocity command hold window.
BASE_HOLD_PADDING: float = 0.05

# Named torso joints that are recorded (JointGroupPositionCommandBuilder subset).
TORSO_JOINT_NAMES: List[str] = ["torso_2", "torso_3"]

# Gripper: maximum range in radians as recorded in simulation.
GRIPPER_MAX_RAD: float = 0.8

# Log a progress line every N steps.
LOG_EVERY_N: int = 20

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_shutdown_requested = False


def _sigint_handler(sig, frame):  # noqa: ANN001
    global _shutdown_requested
    log.warning("Interrupt received — will stop cleanly after current step.")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _sigint_handler)

# ──────────────────────────────────────────────────────────────────────────────
# Trajectory loading
# ──────────────────────────────────────────────────────────────────────────────

def load_trajectories(traj_dir: Path):
    """Load the three parallel pickle files from *traj_dir*."""
    base_path  = traj_dir / "base_trajectories_relative.pkl"
    arm_path   = traj_dir / "arm_trajectories.pkl"
    torso_path = traj_dir / "torso_trajectories.pkl"

    for p in (base_path, arm_path, torso_path):
        if not p.exists():
            raise FileNotFoundError(f"Trajectory file not found: {p}")

    with open(base_path,  "rb") as f:
        base_trajs = pickle.load(f)
    with open(arm_path,   "rb") as f:
        arm_trajs = pickle.load(f)
    with open(torso_path, "rb") as f:
        torso_trajs = pickle.load(f)

    log.info(
        "Loaded %d trajectories from %s  "
        "(base=%d, arm=%d, torso=%d)",
        len(base_trajs), traj_dir,
        len(base_trajs), len(arm_trajs), len(torso_trajs),
    )
    return base_trajs, arm_trajs, torso_trajs

# ──────────────────────────────────────────────────────────────────────────────
# Robot connection & initialisation
# ──────────────────────────────────────────────────────────────────────────────

def connect_robot(address: str):
    """Connect, power on, servo on, and enable the control manager."""
    log.info("Connecting to %s …", address)
    robot = rby.create_robot(address, "a")

    if not robot.connect():
        raise RuntimeError(f"Failed to connect to robot at {address}")
    log.info("Connected.")

    if not robot.is_power_on(".*"):
        log.info("Powering on …")
        if not robot.power_on(".*"):
            raise RuntimeError("power_on('.*') failed")

    if not robot.is_servo_on(".*"):
        log.info("Servo on …")
        if not robot.servo_on(".*"):
            raise RuntimeError("servo_on('.*') failed")

    cm = robot.get_control_manager_state()
    fault_states = (
        rby.ControlManagerState.State.MinorFault,
        rby.ControlManagerState.State.MajorFault,
    )
    if cm.state in fault_states:
        log.warning("Control manager fault detected — resetting …")
        if not robot.reset_fault_control_manager():
            raise RuntimeError("reset_fault_control_manager() failed")

    if not robot.enable_control_manager():
        raise RuntimeError("enable_control_manager() failed")

    # Smooth the command stream with a low-pass filter.
    robot.set_parameter("joint_position_command.cutoff_frequency", "5")
    robot.set_parameter("default.acceleration_limit_scaling", "1.0")

    log.info("Robot ready.")
    return robot


def print_joint_table(robot) -> None:
    """Print joint-index ↔ joint-name mapping for verification."""
    info = robot.get_robot_info()
    names = info.joint_names
    print(f"\n{'Index':<6}  Joint Name")
    print("─" * 35)
    for idx, name in enumerate(names):
        marker = " ◄ torso recorded" if name in TORSO_JOINT_NAMES else ""
        print(f"{idx:<6}  {name}{marker}")
    print()

# ──────────────────────────────────────────────────────────────────────────────
# Gripper controller (DynamixelBus)
# ──────────────────────────────────────────────────────────────────────────────

class GripperController:
    """
    Map simulation radian targets to DynamixelBus encoder positions.

    Homing drives both servos to their mechanical stops to discover the
    encoder range, then switches to current-based position control mode.
    Simulation radians are mapped linearly: 0 rad → open end, GRIPPER_MAX_RAD → closed end.
    """

    _IDS: List[int] = [0, 1]

    def __init__(self) -> None:
        self._bus: Optional[rby.DynamixelBus] = None
        self._min_enc = np.array([np.inf,  np.inf])
        self._max_enc = np.array([-np.inf, -np.inf])
        self._ready = False

    # ------------------------------------------------------------------
    # Init / homing
    # ------------------------------------------------------------------

    def init(self) -> bool:
        log.info("Initialising gripper (DynamixelBus) …")
        self._bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
        self._bus.open_port()
        self._bus.set_baud_rate(2_000_000)
        self._bus.set_torque_constant([1, 1])

        for dev_id in self._IDS:
            if not self._bus.ping(dev_id):
                log.error("Gripper Dynamixel ID %d did not respond to ping.", dev_id)
                return False
            log.info("  Gripper servo %d OK.", dev_id)

        log.info("Homing gripper (≈ 4 s) …")
        self._home()

        # Switch to current-based position mode for compliant control.
        self._bus.group_sync_write_torque_enable([(i, 0) for i in self._IDS])
        self._bus.group_sync_write_operating_mode(
            [(i, rby.DynamixelBus.CurrentBasedPositionControlMode) for i in self._IDS]
        )
        self._bus.group_sync_write_torque_enable([(i, 1) for i in self._IDS])
        self._bus.group_sync_write_send_torque([(i, 5) for i in self._IDS])

        self._send_norm([0.0, 0.0])  # open
        self._ready = True
        log.info(
            "Gripper ready.  enc_min=%s  enc_max=%s",
            self._min_enc.astype(int), self._max_enc.astype(int),
        )
        return True

    def _home(self) -> None:
        """Drive to both mechanical stops; record min/max encoder values."""
        self._bus.group_sync_write_torque_enable([(i, 0) for i in self._IDS])
        self._bus.group_sync_write_operating_mode(
            [(i, rby.DynamixelBus.CurrentControlMode) for i in self._IDS]
        )
        self._bus.group_sync_write_torque_enable([(i, 1) for i in self._IDS])

        prev_q = np.full(2, np.nan)
        for sign in (1.0, -1.0):
            self._bus.group_sync_write_send_torque(
                [(i, 0.5 * sign) for i in self._IDS]
            )
            stuck = 0
            while stuck < 30:
                rv = self._bus.group_fast_sync_read_encoder(self._IDS)
                if rv is not None:
                    q = np.array([enc for _, enc in sorted(rv)])
                    self._min_enc = np.minimum(self._min_enc, q)
                    self._max_enc = np.maximum(self._max_enc, q)
                    stuck = stuck + 1 if np.allclose(prev_q, q, atol=1.0) else 0
                    prev_q = q.copy()
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def send(self, rad_0: float, rad_1: float) -> None:
        """Send gripper targets given simulation joint values in radians."""
        if not self._ready:
            return
        n0 = float(np.clip(rad_0 / GRIPPER_MAX_RAD, 0.0, 1.0))
        n1 = float(np.clip(rad_1 / GRIPPER_MAX_RAD, 0.0, 1.0))
        self._send_norm([n0, n1])

    def _send_norm(self, norms: List[float]) -> None:
        enc_range = self._max_enc - self._min_enc
        targets = [
            int(norms[i] * enc_range[i] + self._min_enc[i]) for i in range(2)
        ]
        self._bus.group_sync_write_send_position(list(zip(self._IDS, targets)))

    def open(self) -> None:
        """Fully open the gripper."""
        self._send_norm([0.0, 0.0])

    def stop(self) -> None:
        """Open gripper and disable servo torque."""
        if self._bus is not None and self._ready:
            try:
                self.open()
                time.sleep(0.3)
                self._bus.group_sync_write_torque_enable([(i, 0) for i in self._IDS])
            except Exception as exc:  # noqa: BLE001
                log.warning("Gripper stop error (ignored): %s", exc)

# ──────────────────────────────────────────────────────────────────────────────
# Command builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_body_cmd(
    left_arm:  np.ndarray,  # (7,)  — absolute joint angles [rad]
    torso:     np.ndarray,  # (2,)  — [torso_2, torso_3] [rad]
    move_time: float,
) -> rby.RobotCommandBuilder:
    """
    Build a left-arm + partial-torso JointPositionCommand.

    Uses JointGroupPositionCommandBuilder for torso so that only torso_2 and
    torso_3 are overridden (the other torso joints are unaffected).
    """
    header = rby.CommandHeaderBuilder().set_control_hold_time(BODY_HOLD_TIME)
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(
                rby.JointGroupPositionCommandBuilder()
                .set_joint_names(TORSO_JOINT_NAMES)
                .set_command_header(header)
                .set_minimum_time(move_time)
                .set_position(torso.tolist())
            )
            .set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_command_header(header)
                .set_minimum_time(move_time)
                .set_position(left_arm.tolist())
            )
        )
    )


def _build_base_cmd(
    vx: float,
    vy: float,
    omega: float,
    hold_time: float,
) -> rby.RobotCommandBuilder:
    """
    Build an SE2 velocity command for the mobile base.

    vx, vy, omega must already be in the **world frame**.
    hold_time should be slightly longer than one step so the base does not
    coast to zero between commands.
    """
    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_mobility_command(
            rby.SE2VelocityCommandBuilder()
            .set_command_header(
                rby.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_velocity([vx, vy], omega)
        )
    )

# ──────────────────────────────────────────────────────────────────────────────
# Go-home helper
# ──────────────────────────────────────────────────────────────────────────────

def go_home(robot, minimum_time: float = 5.0) -> bool:
    """Command all 20 body joints to zero (torso + right arm + left arm)."""
    log.info("Moving to zero pose (%.1f s) …", minimum_time)
    rv = robot.send_command(
        rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(
                rby.JointPositionCommandBuilder()
                .set_position(np.zeros(20))
                .set_minimum_time(minimum_time)
            )
        ),
        1,
    ).get()
    ok = rv.finish_code == rby.RobotCommandFeedback.FinishCode.Ok
    if not ok:
        log.error("go_home finished with: %s", rv.finish_code)
    return ok

# ──────────────────────────────────────────────────────────────────────────────
# Single-trajectory rollout
# ──────────────────────────────────────────────────────────────────────────────

def rollout(
    robot,
    stream,
    base_traj:     np.ndarray,              # (T, 3)  — [Δx, Δy, Δθ] local frame
    left_arm_traj: np.ndarray,              # (T, 7)  — absolute joint angles [rad]
    torso_traj:    np.ndarray,              # (T, 2)  — absolute [torso_2, torso_3] [rad]
    gripper_traj:  np.ndarray,              # (T, 2)  — absolute gripper targets [rad]
    gripper_ctrl:  Optional[GripperController],
    speed_scale:   float = 1.0,
    skip_base:     bool  = False,
) -> None:
    """
    Replay one recorded trajectory on the real RBY1.

    Body (arm + torso) commands are sent via the persistent command stream.
    Base deltas are converted from the robot's local frame to world-frame
    velocities using the accumulated heading estimate.
    The step timing loop busy-waits to maintain the recorded rate.
    """
    global _shutdown_requested

    T = len(base_traj)
    assert len(left_arm_traj) == T, "arm / base length mismatch"
    assert len(torso_traj)    == T, "torso / base length mismatch"
    assert len(gripper_traj)  == T, "gripper / base length mismatch"

    # Slower speed_scale stretches the wall-clock interval per step.
    step_dt   = STEP_DT / speed_scale
    move_time = step_dt * MOVE_TIME_FACTOR
    base_hold = step_dt + BASE_HOLD_PADDING

    current_yaw = 0.0  # accumulated heading estimate (rad)
    over_budget_count = 0

    log.info(
        "Rollout start: %d steps  step_dt=%.3f s  speed_scale=%.2f  "
        "skip_base=%s  gripper=%s",
        T, step_dt, speed_scale,
        skip_base, gripper_ctrl is not None,
    )

    for t in range(T):
        if _shutdown_requested:
            log.warning("Interrupted at step %d / %d.", t, T)
            break

        t0 = time.perf_counter()

        # ── 1. Arm + torso ─────────────────────────────────────────────
        body_cmd = _build_body_cmd(left_arm_traj[t], torso_traj[t], move_time)
        stream.send_command(body_cmd)

        # ── 2. Mobile base ─────────────────────────────────────────────
        if not skip_base:
            dx, dy, dtheta = (
                float(base_traj[t, 0]) * speed_scale,
                float(base_traj[t, 1]) * speed_scale,
                float(base_traj[t, 2]) * speed_scale,
            )
            # Rotate local-frame displacements into the world frame.
            cos_y = math.cos(current_yaw)
            sin_y = math.sin(current_yaw)
            vx    = (dx * cos_y - dy * sin_y) / step_dt
            vy    = (dx * sin_y + dy * cos_y) / step_dt
            omega = dtheta / step_dt
            current_yaw += dtheta

            base_cmd = _build_base_cmd(vx, vy, omega, base_hold)
            stream.send_command(base_cmd)

        # ── 3. Gripper ─────────────────────────────────────────────────
        if gripper_ctrl is not None:
            gripper_ctrl.send(float(gripper_traj[t, 0]), float(gripper_traj[t, 1]))

        # ── 4. Timing ──────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        sleep_t = step_dt - elapsed
        if sleep_t > 1e-4:
            time.sleep(sleep_t)
        else:
            over_budget_count += 1
            if t % LOG_EVERY_N == 0:
                log.warning(
                    "Step %d: over budget by %.1f ms  (total over-budget: %d)",
                    t, -sleep_t * 1e3, over_budget_count,
                )

        if t % LOG_EVERY_N == 0 or t == T - 1:
            log.info("  step %d / %d  (yaw=%.3f rad)", t + 1, T, current_yaw)

    if over_budget_count > 0:
        log.warning(
            "Rollout finished: %d / %d steps exceeded budget.", over_budget_count, T
        )
    else:
        log.info("Rollout complete — all steps within timing budget.")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay RBY1 whole-body trajectories on the real robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--address", default="192.168.1.1:50051",
        help="Robot gRPC address (host:port)",
    )
    p.add_argument(
        "--traj-dir", default="trajectories",
        help="Directory containing the three .pkl trajectory files",
    )
    p.add_argument(
        "--traj-index", type=int, default=None,
        help="Run only this trajectory index (default: run all)",
    )
    p.add_argument(
        "--speed-scale", type=float, default=1.0,
        help="Speed multiplier (e.g. 0.25 for first cautious run)",
    )
    p.add_argument(
        "--step-dt", type=float, default=STEP_DT,
        help="Physics step interval in seconds — must match recording",
    )
    p.add_argument(
        "--skip-base", action="store_true",
        help="Do not command the mobile base (arm + torso only)",
    )
    p.add_argument(
        "--enable-gripper", action="store_true",
        help="Enable DynamixelBus gripper control (requires wired UPC gripper)",
    )
    p.add_argument(
        "--go-home", action="store_true",
        help="Command all body joints to zero before each trajectory",
    )
    p.add_argument(
        "--home-time", type=float, default=5.0,
        help="Time (s) to reach the zero / home pose",
    )
    p.add_argument(
        "--no-confirm", action="store_true",
        help="Skip the per-trajectory Enter-to-start prompt",
    )
    p.add_argument(
        "--print-joints", action="store_true",
        help="Print joint-index table and exit (no motion)",
    )
    return p.parse_args()


def main() -> None:
    global STEP_DT
    args = parse_args()
    STEP_DT = args.step_dt  # allow CLI override

    # ── Connect ────────────────────────────────────────────────────────
    robot = connect_robot(args.address)

    if args.print_joints:
        print_joint_table(robot)
        robot.disconnect()
        return

    if args.speed_scale != 1.0:
        log.info(
            "Speed scale %.2f → effective step_dt = %.3f s  (nominal %.3f s)",
            args.speed_scale, STEP_DT / args.speed_scale, STEP_DT,
        )

    # ── Gripper (optional) ─────────────────────────────────────────────
    gripper_ctrl: Optional[GripperController] = None
    if args.enable_gripper:
        gripper_ctrl = GripperController()
        if not gripper_ctrl.init():
            log.error("Gripper initialisation failed — aborting.")
            robot.disconnect()
            sys.exit(1)

    # ── Load trajectory files ──────────────────────────────────────────
    traj_dir = Path(args.traj_dir)
    base_trajs, arm_trajs, torso_trajs = load_trajectories(traj_dir)

    n_total = len(base_trajs)
    indices = [args.traj_index] if args.traj_index is not None else list(range(n_total))
    log.info("Will replay %d trajectory/trajectories.", len(indices))

    # ── Create a persistent command stream (priority 1) ────────────────
    # The stream lets us push commands at the recorded frequency without
    # the overhead of opening a new RPC per step.
    stream = robot.create_command_stream(1)

    # ── Main loop ──────────────────────────────────────────────────────
    try:
        for run_num, traj_idx in enumerate(indices):
            if _shutdown_requested:
                log.info("Shutdown requested — stopping before trajectory %d.", traj_idx)
                break

            log.info(
                "\n=== Trajectory %d / %d  (dataset index %d) ===",
                run_num + 1, len(indices), traj_idx,
            )

            base_traj  = np.asarray(base_trajs[traj_idx],  dtype=np.float64)
            arm_traj   = np.asarray(arm_trajs[traj_idx],   dtype=np.float64)
            torso_traj = np.asarray(torso_trajs[traj_idx], dtype=np.float64)

            left_arm_traj = arm_traj[:, :7]   # (T, 7)
            gripper_traj  = arm_traj[:, 7:9]  # (T, 2)

            T = len(base_traj)
            log.info(
                "  Steps: %d   Nominal duration: %.1f s",
                T, T * STEP_DT / args.speed_scale,
            )

            if not args.no_confirm:
                try:
                    input("\n  Press Enter to start this trajectory (Ctrl-C to abort) … ")
                except EOFError:
                    pass  # non-interactive — just proceed

            if args.go_home:
                if not go_home(robot, minimum_time=args.home_time):
                    log.error(
                        "go_home failed for trajectory %d — skipping.", traj_idx
                    )
                    continue

            rollout(
                robot, stream,
                base_traj      = base_traj,
                left_arm_traj  = left_arm_traj,
                torso_traj     = torso_traj,
                gripper_traj   = gripper_traj,
                gripper_ctrl   = gripper_ctrl,
                speed_scale    = args.speed_scale,
                skip_base      = args.skip_base,
            )

    finally:
        log.info("Cleaning up …")
        if gripper_ctrl is not None:
            gripper_ctrl.stop()
        robot.cancel_control()
        robot.disconnect()
        log.info("Disconnected.")


if __name__ == "__main__":
    main()
