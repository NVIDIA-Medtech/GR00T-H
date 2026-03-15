# USTC Surgical Dataset (Torin)

## Overview

USTC provides surgical demonstrations recorded on the Torin platform with stereo endoscope video. The merged dataset contains seven task families: `exvivo_liver_sep`, `grasp_on_liver`, `invivo_liver_sep`, `knot_tying`, `Needle_handover`, `needle_pickup`, and `tissue_lifting`. All subsets are in **LeRobot v2.1** format.

### Note

This integration uses `action.cartesian_absolute` (xyz + quaternion + gripper per arm)
for pose targets and REL_XYZ_ROT6D for action modeling. Gripper is modeled as an
absolute target. `action.cartesian_absolute` is written alongside `absolute_action`
during conversion, and invalid rotations are repaired by carrying forward the last
valid rotation (identity if none).

## Embodiment Configuration

| Property | Value |
|---------|-------|
| Embodiment Tag | `ustc_torin_tuodao` |
| Config File | `open_h/embodiments/ustc_torin_tuodao/ustc_torin_tuodao_config.py` |
| Modality Mapping | `open_h/embodiments/ustc_torin_tuodao/modality.json` |
| Action Horizon | 50 (delta_indices `[0..49]`, 24 Hz) |

## Tasks and Subsets

### Subset Statistics (merged)

| Subset | Episodes | Frames | FPS | Hours |
|--------|----------|--------|-----|-------|
| exvivo_liver_sep | 666 | 121,922 | 24 | 1.41 |
| grasp_on_liver | 817 | 63,538 | 24 | 0.74 |
| invivo_liver_sep | 199 | 39,899 | 24 | 0.46 |
| knot_tying | 1,098 | 182,836 | 24 | 2.12 |
| Needle_handover | 260 | 34,990 | 24 | 0.41 |
| needle_pickup | 616 | 57,172 | 24 | 0.66 |
| tissue_lifting | 110 | 11,673 | 24 | 0.14 |

### Task Totals

| Task | Episodes | Frames | FPS | Hours |
|------|----------|--------|-----|-------|
| **Overall** | **3,766** | **512,030** | **24** | **5.94** |

## Data Format

### Cameras

| Camera Key | Modality Key | Resolution | Codec | FPS | Note |
|-----------|-------------|------------|-------|-----|------|
| `observation.images.endoscope.left` | `endoscope_left` | 1080x1920 | H.264 | 24 | Active in config |
| `observation.images.endoscope.right` | `endoscope_right` | 1080x1920 | H.264 | 24 | Available in modality.json but commented out in config (mono only) |

### Kinematics

**State (`observation.state`, 14D):** Joint angles for both arms.

- Left arm: `left_joint_1` ... `left_joint_7`
- Right arm: `right_joint_8` ... `right_joint_14`

**Action (`action`, 14D):** Cartesian **delta** commands per arm using Euler angles (plus gripper).

- Left arm: `left_x`, `left_y`, `left_z`, `left_roll`, `left_pitch`, `left_yaw`, `left_gripper`
- Right arm: `right_x`, `right_y`, `right_z`, `right_roll`, `right_pitch`, `right_yaw`, `right_gripper`

**Absolute Pose (`action.cartesian_absolute`, 16D):** Cartesian absolute pose per arm
(xyz + quaternion + gripper).

- Left arm: `left_x`, `left_y`, `left_z`, `left_qx`, `left_qy`, `left_qz`, `left_qw`, `left_gripper`
- Right arm: `right_x`, `right_y`, `right_z`, `right_qx`, `right_qy`, `right_qz`, `right_qw`, `right_gripper`

**Current Pose (`observation.current_target_psm`, 16D):** Cartesian absolute pose per arm
(xyz + quaternion + gripper) at time t.

**Energy (`energy`, 1D):** Not present in the current USTC parquets; a future update
should append energy into the action vector once regenerated.

### Other Metadata

- `instruction.text` (string)
- `observation.meta.tool` (string, `left_right`)
- `timestamp`, `frame_index`, `episode_index`, `task_index`

## Modality Keys (config.py)

**State keys (tokenized):** `left_joints` (7D), `right_joints` (7D)
**Pass-through keys:** `left_pose` (7D from `observation.current_target_psm[0:7]`), `right_pose` (7D from `observation.current_target_psm[8:15]`) -- used as REL_XYZ_ROT6D reference frames, not tokenized as state.

**Action keys:**
- `left_pose` (7D, REL_XYZ_ROT6D from `action.cartesian_absolute[0:7]`)
- `left_gripper` (1D, ABSOLUTE from `action.cartesian_absolute[7:8]`)
- `right_pose` (7D, REL_XYZ_ROT6D from `action.cartesian_absolute[8:15]`)
- `right_gripper` (1D, ABSOLUTE from `action.cartesian_absolute[15:16]`)

**Video key:** `endoscope_left`
**Language key:** `annotation.instruction` (maps to `instruction.text`, raw text strings)

## Episode Length Check (50-Step Horizon)

All subsets are 24 Hz (50 steps ≈ 2.08 seconds). Only one episode in `knot_tying/2` has length 2; every other episode across the dataset is at least 54 steps long. The dataloader automatically skips episodes that are too short for the requested action horizon, so no manual removal is required.

## Dataset Preparation

Use the shared preparation script to copy the modality JSON into each dataset's `meta/` folder and generate normalization statistics:

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag ustc_torin_tuodao \
    --modality-json open_h/embodiments/ustc_torin_tuodao/modality.json \
    /path/to/dataset
```

## Integration Notes

- `observation.state` contains joint angles only; Cartesian EEF pose is sourced from `action.cartesian_absolute`.
- Pose actions use REL_XYZ_ROT6D conversion with the reference pose from `observation.current_target_psm`.
- Gripper is modeled as an ABSOLUTE action and appended after pose keys.
- Language uses per-frame `instruction.text` strings (mapped via `annotation.instruction`).
- **Warning: Data quality consideration** — Some subsets have all-zero rotation matrices in the raw data for one arm; the LeRobot conversion script
  used replaced invalid rotations with the previous valid rotation (identity if none).
