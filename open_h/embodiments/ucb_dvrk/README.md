# UCBerkeley dVRK

Debridement dataset collected from the da Vinci Research Kit (dVRK) with stereo endoscope views and cartesian EEF pose actions (REL_XYZ_ROT6D).

## Dataset Statistics

| Property | Value |
|----------|-------|
| **Episodes** | 589 |
| **Total Frames** | 221,950 |
| **Avg Frames/Episode** | ~377 |

## Embodiment Configuration

| Property | Value |
|----------|-------|
| **Embodiment Tag** | `ucb_dvrk` |
| **Projector Index** | 5 |
| **Config File** | `open_h/embodiments/ucb_dvrk/ucb_dvrk_config.py` |
| **Action Horizon** | 50 (delta_indices `[0..49]`) |

## Data Format

### State (Cartesian + Joints)
Cartesian EEF state for both PSM arms plus joint angles:

| Arm | Indices | Dimensions | Description |
|-----|---------|------------|-------------|
| PSM1 pose | 0-6 | 7D | `[x, y, z, qx, qy, qz, qw]` |
| PSM1 gripper | 7 | 1D | `jaw` |
| PSM2 pose | 8-14 | 7D | Same format as PSM1 pose |
| PSM2 gripper | 15 | 1D | `jaw` |

Joint-angle state (sourced from `observation.state`):
| Arm | Indices | Dimensions | Description |
|-----|---------|------------|-------------|
| PSM1 joints | 0-6 | 7D | `[outer_yaw, outer_pitch, outer_insertion, outer_roll, outer_wrist_pitch, outer_wrist_yaw, jaw]` |
| PSM2 joints | 7-13 | 7D | Same format as PSM1 |

**Pass-through keys:** `psm1_pose`, `psm2_pose` (used as REL_XYZ_ROT6D reference frames, not tokenized as state).
**Model state keys (tokenized):** `psm1_joints`, `psm1_gripper`, `psm2_joints`, `psm2_gripper`.

### Action (16D)
Same format as state, representing cartesian setpoints. Converted to **REL_XYZ_ROT6D** during training:
- `action[t] == state[t+1]` (setpoint for the next timestep)
- Relative conversion uses state[t] as the reference frame for the action horizon

### Video (1 view used for training)
| Camera | Original Key | Note |
|--------|--------------|------|
| `camera_left` | `observation.images.left` | Active in config |
| `camera_right` | `observation.images.right` | Available in modality.json but commented out in config (mono only) |

### Language
| Key |
|-----|
| `task` |

## Dataset Preparation

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag UCB_DVRK \
    --modality-json open_h/embodiments/ucb_dvrk/modality.json \
    /path/to/ucb_dvrk_dataset
```

This generates `stats.json` and `temporal_stats.json` inside the dataset meta folder.


## File Structure

```
open_h/embodiments/ucb_dvrk/
├── README.md            # This file
├── ucb_dvrk_config.py   # Built-in config module (auto-registered)
└── modality.json        # Data key mappings (copied to dataset/meta/ by script)
```
