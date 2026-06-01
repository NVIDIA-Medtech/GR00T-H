# TUD TUNDRA (UR5e) - Open-H Embodiment

TUD Dresden teleoperation dataset for surgical assistance using a Universal Robots UR5e with stereo laparoscope.

## Embodiment Configuration

| Property | Value |
|----------|-------|
| **Embodiment Tag** | `TUD_TUNDRA_UR5E` |
| **Config File** | `open_h/embodiments/tud_tundra_ur5e/tud_tundra_ur5e_config.py` |

## Data Summary

| Subset | Episodes | FPS | Task |
|--------|----------|-----|------|
| `grasping_retraction` | 146 | 30 Hz | Grasping and tissue retraction during in-vivo porcine surgery |

## Data Format

### State

- `joint_position` (6) -- embedded via mean/std
- `eef_pose` (7) -- base-frame XYZ + quaternion, pass-through reference for REL_XYZ_ROT6D

### Action

- `eef_pose` (7) -- absolute EEF pose from `observation.state` at future timesteps (t+1..t+H)
- `gripper` (1) -- from `open_gripper` (`action[4]`), binary

Action representation: `REL_XYZ_ROT6D` for EEF pose, `ABSOLUTE` for gripper.
Action horizon: 50 (~1.7s at 30 Hz). `delta_indices = 1..H`.

The dataset action column contains delta commands, but REL_XYZ_ROT6D uses absolute
EEF poses from `observation.state` with a +1 offset for action timesteps.

### Cameras

- `laparoscope_left` (960x540 @ 30 Hz)
- Note: `laparoscope_right` is available in the dataset but only the left view is used in the config (mono only).

### Language

- `task` -- mapped from `task_index` via `tasks.jsonl`

## Modality Mapping

- `modality_grasping_retraction.json`

## Dataset Preparation

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag TUD_TUNDRA_UR5E \
    --modality-json open_h/embodiments/tud_tundra_ur5e/modality_grasping_retraction.json \
    /path/to/tud_dataset
```

## Notes

- Endoscope guidance is not configured yet; only grasping_retraction is supported in this config.
- Endoscope guidance can be included with a small additional config that drops the gripper action, adding ~50 episodes to the TUD dataset.
- The `gripper_opened` state value is constant (0) in grasping_retraction; use the `open_gripper` action (`action[4]`) as the only valid gripper signal.
- The action pipeline uses `delta_indices = 1..H`. However, the gripper is sourced from `action` while the pose is sourced from `observation.state` where `action[t] = state[t+1]`, so there is some temporal misalignment for the gripper.
