# Moon Surgical (Maestro) Embodiment

Dual-arm laparoscopic assistant robot with 20 task-level commands covering scope maintenance, anatomy navigation, instrument tracking, and zoom.

## Embodiment Configuration

| Property | Value |
|----------|-------|
| **Embodiment Tag** | `MOON_MAESTRO` |
| **Config File** | `open_h/embodiments/moon_maestro/moon_maestro_config.py` |
| **Modality Mapping** | `open_h/embodiments/moon_maestro/modality.json` |

## Data Summary

| Property | Value |
|----------|-------|
| Source | Moon Surgical (Maestro laparoscopic assistant) |
| Tasks | 20 assistant commands |
| Episodes | 65 |
| Frames | 12,020 |
| FPS | 30 |

## Data Format

### State
- 18D joint positions (9 joints per arm)
  - `right_arm_joints`: joint_0_arm_0 .. joint_8_arm_0
  - `left_arm_joints`: joint_0_arm_1 .. joint_8_arm_1

### Action
- 6D delta translation in base frame
  - `right_arm_delta_xyz`: d_tx_arm_0, d_ty_arm_0, d_tz_arm_0
  - `left_arm_delta_xyz`: d_tx_arm_1, d_ty_arm_1, d_tz_arm_1
- Action representation in config: `DELTA` + `ActionFormat.XYZ`

### Cameras
- `scope`: 960x540
- `topcam`: 1280x720

### Language
- `tasks.jsonl` via `annotation.task` mapping in `modality.json`

## Action Horizon
- 30 Hz, `ACTION_HORIZON = 50` (1.67 s prediction)

## Dataset Preparation

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag MOON_MAESTRO \
    --modality-json open_h/embodiments/moon_maestro/modality.json \
    /path/to/moon_maestro_dataset
```

Note: Stats generation and the load test require `meta/modality.json` in the dataset path.

## Notes
- No end-effector pose in the raw dataset (joint-only state).
- Actions are translation-only deltas (no rotation, no gripper signals).
