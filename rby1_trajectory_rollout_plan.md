# Rolling Out Recorded Whole-Body Trajectories on the Real RBY1

This document describes how to take the pickle files produced by
`record_whole_body_trajectories.py` and replay them on a physical
Rainbow Robotics RBY1 using the `rby1-sdk`.

---

## 1. What the Pickle Files Contain

Three parallel lists are saved, one entry per recorded demonstration:

| File (default path) | Variable | Shape per entry | Meaning |
|---|---|---|---|
| `base_trajectories_relative.pkl` | `base_traj` | `(T, 3)` float32 | Per-step **relative** base deltas `[Δx, Δy, Δθ]` in the robot's **local** frame at that instant |
| `arm_trajectories.pkl` | `arm_traj` | `(T, 9)` float32 | Columns 0–6: left-arm joint angles (rad); columns 7–8: gripper joint targets |
| `torso_trajectories.pkl` | `torso_traj` | `(T, 2)` float32 | `[torso_2, torso_3]` joint angles (rad) |

`T` is the number of timesteps in that trajectory (varies per demo).
A fourth list `trajectory_metadatas` carries per-demo dicts (start pose,
object/obstacle pose, planning provenance, etc.) — not needed for pure
playback but useful for logging.

---

## 2. Loading the Files

```python
import pickle
import numpy as np

def load_trajectories(base_path, arm_path, torso_path):
    with open(base_path,  "rb") as f: base_trajs  = pickle.load(f)
    with open(arm_path,   "rb") as f: arm_trajs   = pickle.load(f)
    with open(torso_path, "rb") as f: torso_trajs = pickle.load(f)
    return base_trajs, arm_trajs, torso_trajs

base_trajs, arm_trajs, torso_trajs = load_trajectories(
    "trajectories/base_trajectories_relative.pkl",
    "trajectories/arm_trajectories.pkl",
    "trajectories/torso_trajectories.pkl",
)
```

Each element is a `(T, D)` numpy array. Index them together:

```python
traj_idx = 0
base_traj  = np.asarray(base_trajs[traj_idx],  dtype=np.float32)  # (T, 3)
arm_traj   = np.asarray(arm_trajs[traj_idx],   dtype=np.float32)  # (T, 9)
torso_traj = np.asarray(torso_trajs[traj_idx], dtype=np.float32)  # (T, 2)

left_arm_traj = arm_traj[:, :7]   # (T, 7) — joint angles in radians
gripper_traj  = arm_traj[:, 7:9]  # (T, 2) — gripper joint targets
```

---

## 3. Trajectory Semantics

### 3.1 Base — relative local-frame deltas

`override_base_action` in the sim shows the exact convention:

```
Δx_world = Δx_local * cos(yaw) - Δy_local * sin(yaw)
Δy_world = Δx_local * sin(yaw) + Δy_local * cos(yaw)
yaw_new  = yaw + Δθ
```

So each row `[Δx, Δy, Δθ]` is a **body-frame** displacement. On the real
robot you apply it to whatever odometry / pose estimate you maintain.

### 3.2 Left arm — absolute joint angles

`left_arm_traj[t]` is the **target joint position** (rad) for the 7 left-arm
joints, in this order (matching the sim's `left_arm_0 … left_arm_6`):

```
left_arm_0, left_arm_1, left_arm_2,
left_arm_3, left_arm_4, left_arm_5, left_arm_6
```

Verify the mapping against `rby1_sdk` joint ordering before running.

### 3.3 Gripper — absolute joint targets

`gripper_traj[t]` → `[gripper_l_joint, gripper_l_joint_m]` positions (rad).

### 3.4 Torso — absolute joint angles

`torso_traj[t]` → `[torso_2, torso_3]` positions (rad).

---

## 4. Timing

The simulation was run at a fixed physics step. You need to know the
**step_dt** used during recording (check `env_cfg.episode_length_s` and the
number of steps, or look in the trajectory metadata). A typical value is
**50 ms per step (20 Hz)**. Use that as the wall-clock interval between
commands on the real robot.

```python
STEP_DT = 0.05  # seconds — adjust to match your sim step_dt
```

---

## 5. RBY1 SDK Setup

```python
import rby1_sdk as rby
import time, numpy as np

ROBOT_ADDRESS = "192.168.1.1:50051"   # update to your robot's IP

robot = rby.create_robot_a(ROBOT_ADDRESS)
assert robot.connect(), "Failed to connect"
print(robot.get_robot_info())
```

---

## 6. Joint Name → SDK Index Mapping

Model A full joint vector layout (0-indexed):

| Segment | Joints | Indices |
|---|---|---|
| Base / mobility | — | controlled separately via base API |
| Torso | torso_0 … torso_3 | 3 – 6 |
| Right arm | right_arm_0 … right_arm_6 | 7 – 13 |
| Left arm | left_arm_0 … left_arm_6 | 14 – 20 |
| Head | head_0, head_1 | 21 – 22 |

The recorded trajectories use:
- **Left arm**: indices 14–20
- **Torso**: torso_2 = index 5, torso_3 = index 6
- **Gripper**: check `robot.get_robot_info()` for the exact gripper joint indices; typically appended after the arm joints.

Confirm at runtime:
```python
info = robot.get_robot_info()
for i, j in enumerate(info.joint_infos):
    print(i, j.name)
```

---

## 7. Rollout Loop

```python
import rby1_sdk as rby
import numpy as np
import time

ROBOT_ADDRESS = "192.168.1.1:50051"
STEP_DT       = 0.05   # seconds — must match recording step_dt
MOVE_TIME     = STEP_DT * 1.5  # give the controller a little headroom

# --- joint index constants (verify against your firmware) ---
LEFT_ARM_INDICES   = list(range(14, 21))  # left_arm_0 … left_arm_6
TORSO_INDICES      = [5, 6]               # torso_2, torso_3
GRIPPER_INDICES    = [21, 22]             # update after checking joint list


def send_joint_targets(robot, positions: dict, move_time: float):
    """
    positions: {joint_index: angle_rad, ...}
    Sends a JointPositionCommand for the specified joints.
    """
    indices = list(positions.keys())
    angles  = [positions[i] for i in indices]

    cmd = (
        rby.RobotCommandBuilder()
        .set_command(
            rby.ComponentBasedCommandBuilder()
            .set_body_command(
                rby.BodyComponentBasedCommandBuilder()
                .set_arm_command(
                    rby.ArmCommandBuilder()
                    .set_command(
                        rby.JointPositionCommandBuilder()
                        .set_command_header(
                            rby.CommandHeaderBuilder()
                            .set_control_hold_time(move_time + 1.0)
                        )
                        .set_minimum_time(move_time)
                        .set_position(angles)
                    ),
                    "left",
                )
            )
        )
    )
    handler = robot.send_command(cmd, 1)
    return handler


def apply_base_delta(robot, delta_x, delta_y, delta_theta, current_yaw):
    """
    Convert a local-frame delta to a world-frame velocity command and send it.
    Uses the RBY1 SE2 velocity command for the mobile base.
    The deltas are converted to velocities by dividing by STEP_DT.
    """
    # Rotate to world frame
    dx_world = delta_x * np.cos(current_yaw) - delta_y * np.sin(current_yaw)
    dy_world = delta_x * np.sin(current_yaw) + delta_y * np.cos(current_yaw)

    vx    = dx_world    / STEP_DT
    vy    = dy_world    / STEP_DT
    omega = delta_theta / STEP_DT

    cmd = (
        rby.RobotCommandBuilder()
        .set_command(
            rby.ComponentBasedCommandBuilder()
            .set_mobility_command(
                rby.MobilityCommandBuilder()
                .set_command(
                    rby.SE2VelocityCommandBuilder()
                    .set_command_header(
                        rby.CommandHeaderBuilder()
                        .set_control_hold_time(STEP_DT + 0.05)
                    )
                    .set_velocity([vx, vy], omega)
                    .set_acceleration_limit([3.0, 3.0], 2.0)
                )
            )
        )
    )
    robot.send_command(cmd, 1)
    return current_yaw + delta_theta


def rollout_trajectory(robot, base_traj, left_arm_traj, torso_traj, gripper_traj):
    """
    Replay one recorded trajectory on the real RBY1.

    Args:
        robot         : connected rby1_sdk robot instance
        base_traj     : (T, 3) float32 — relative base deltas [Δx, Δy, Δθ]
        left_arm_traj : (T, 7) float32 — left arm joint angles (rad)
        torso_traj    : (T, 2) float32 — torso joint angles (rad)
        gripper_traj  : (T, 2) float32 — gripper joint targets (rad)
    """
    T = len(base_traj)
    assert len(left_arm_traj) == T
    assert len(torso_traj)    == T
    assert len(gripper_traj)  == T

    # Read starting yaw from odometry (or set to 0 if not available)
    state       = robot.get_state()
    current_yaw = 0.0  # replace with odometry yaw if available

    print(f"Starting rollout: {T} steps at {1/STEP_DT:.0f} Hz")

    for t in range(T):
        step_start = time.perf_counter()

        # 1. Base: convert delta to velocity and command
        dx, dy, dtheta = base_traj[t]
        current_yaw = apply_base_delta(robot, dx, dy, dtheta, current_yaw)

        # 2. Arm + torso: build a combined joint position command
        #    Map each joint index to its target angle
        joint_targets = {}
        for k, idx in enumerate(LEFT_ARM_INDICES):
            joint_targets[idx] = float(left_arm_traj[t, k])
        for k, idx in enumerate(TORSO_INDICES):
            joint_targets[idx] = float(torso_traj[t, k])
        for k, idx in enumerate(GRIPPER_INDICES):
            joint_targets[idx] = float(gripper_traj[t, k])

        send_joint_targets(robot, joint_targets, move_time=MOVE_TIME)

        # 3. Busy-wait to maintain step rate
        elapsed = time.perf_counter() - step_start
        sleep_t = STEP_DT - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
        elif t % 20 == 0:
            print(f"[WARN] Step {t}: over budget by {-sleep_t*1000:.1f} ms")

        if t % 20 == 0 or t == T - 1:
            print(f"  Step {t+1}/{T}")

    print("Rollout complete.")
```

---

## 8. Running Multiple Trajectories

```python
robot = rby.create_robot_a(ROBOT_ADDRESS)
robot.connect()

base_trajs, arm_trajs, torso_trajs = load_trajectories(
    "trajectories/base_trajectories_relative.pkl",
    "trajectories/arm_trajectories.pkl",
    "trajectories/torso_trajectories.pkl",
)

for traj_idx in range(len(base_trajs)):
    print(f"\n=== Trajectory {traj_idx + 1}/{len(base_trajs)} ===")

    base_traj  = np.asarray(base_trajs[traj_idx],  dtype=np.float32)
    arm_traj   = np.asarray(arm_trajs[traj_idx],   dtype=np.float32)
    torso_traj = np.asarray(torso_trajs[traj_idx], dtype=np.float32)

    left_arm_traj = arm_traj[:, :7]
    gripper_traj  = arm_traj[:, 7:9]

    # Move robot back to a known start pose here if needed, then:
    rollout_trajectory(robot, base_traj, left_arm_traj, torso_traj, gripper_traj)

    input("Press Enter to continue to next trajectory...")

robot.disconnect()
```

---

## 9. Critical Checks Before Running

1. **Verify joint ordering** — print `robot.get_robot_info()` and confirm
   `LEFT_ARM_INDICES`, `TORSO_INDICES`, and `GRIPPER_INDICES` match the
   physical joint names. A mismatch here will cause the wrong joints to move.

2. **Confirm step_dt** — check the `env_cfg` or trajectory metadata for the
   exact physics step used. If the real robot runs faster or slower than the
   recorded rate, motions will be too fast or too slow.

3. **E-stop is accessible** — have someone ready to trigger the hardware
   e-stop during first runs of each new trajectory set.

4. **Slow first run** — consider scaling all deltas and velocities by 0.25–0.5
   on the first playback to verify the trajectory is safe before running at
   full speed.

5. **Base odometry drift** — the base trajectory uses local-frame deltas
   accumulated over `T` steps. Without closed-loop odometry correction,
   positional error will accumulate. For long trajectories consider adding an
   external localization source.

6. **Arm command granularity** — the sim applies joint targets every physics
   step. The real robot's joint controller may have a different control
   frequency. If commands arrive faster than the robot can process them,
   the SDK will queue or drop them. Verify with `robot.get_robot_info()` what
   the control loop rate is and match `STEP_DT` accordingly.

---

## 10. Sim vs Real Correspondence Summary

| Sim variable / function | Real-robot equivalent |
|---|---|
| `env._active_traj[0][t]` → `[Δx, Δy, Δθ]` | `base_traj[t]` fed to `apply_base_delta()` |
| `env._left_arm_trajectory[0][t]` | `left_arm_traj[t]` → `LEFT_ARM_INDICES` |
| `env._gripper_command[0][t]` | `gripper_traj[t]` → `GRIPPER_INDICES` |
| `torso_entity_cfg` joint targets | `torso_traj[t]` → `TORSO_INDICES` |
| `override_base_action` local→world rotation | `apply_base_delta()` does the same math |
| Physics step (`step_dt`) | `STEP_DT` sleep + `set_control_hold_time` |
