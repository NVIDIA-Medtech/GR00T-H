# UCSD Surgical Learning Dataset (dVRK)

## Overview

UCSD surgical learning data in LeRobot v2.1 format with stereo endoscope video,
dVRK kinematics, and two-arm action streams. The datasets include absolute
end-effector (EEF) pose in state and delta EEF actions for retraction and
cutting tools.

## Subsets

| Subset | Episodes | Frames | FPS | Tasks | Min Episode Length |
|--------|----------|--------|-----|-------|--------------------|
| surgical_learning_dataset | 912 | 288,604 | 30 | 2 | 73 |
| surgical_learning_dataset2 | 200 | 26,313 | 30 | 1 | 66 |

All episodes are at least 50 frames, so a 50-step action horizon is valid.

## Task List

- `surgical_learning_dataset`: retraction, dissection
- `surgical_learning_dataset2`: Retraction

## Cameras

Stereo endoscope views:

| Camera | Original Key | Resolution | FPS | Codec | Note |
|--------|--------------|------------|-----|-------|------|
| Left | `observation.images.left` | 480x640 | 30 | AV1 | Active in config (`camera_left`) |
| Right | `observation.images.right` | 480x640 | 30 | AV1 | Available in modality JSON but commented out in config (mono only) |

## Kinematics

### Actions (both datasets)

16D delta EEF pose + gripper for each arm:

- Retraction arm: `dPSM_RETRACTION_x,y,z,qw,qx,qy,qz,gripper`
- Cutter arm: `dPSM_CUTTER_x,y,z,qw,qx,qy,qz,gripper`
- Quaternion order is **wxyz** (scalar-first).

### State

`surgical_learning_dataset` (62D):
- Joint positions, velocities, and efforts for retraction + cutter PSMs
- Gripper position and effort for each tool
- Absolute EEF pose for both arms: `*_ee_x,y,z,qw,qx,qy,qz`
- Target points: `target_0..3_x,y`

`surgical_learning_dataset2` (28D):
- Joint positions + gripper positions for both PSMs
- Absolute EEF pose for both arms: `*_ee_x,y,z,qw,qx,qy,qz`

## Notes

- Absolute EEF state is available for both arms, enabling REL_XYZ_ROT6D action
  conversion using the state at timestep t as reference.
- Quaternion order is **wxyz** (qw, qx, qy, qz) for both state and action.
- `task` entries are available in `tasks.jsonl` and are used for language.

## Embodiment Configuration

| Property | Value |
|----------|-------|
| Embodiment Tag | `ucsd_dvrk` |
| Config File | `open_h/embodiments/ucsd_dvrk/ucsd_dvrk_config.py` |
| Action Horizon | 50 (delta_indices `[1..50]`, 30 Hz) |

## Modality Mapping

The modality JSON filenames intentionally mirror the dataset directory names:
`surgical_learning_dataset` uses
`modality_surgical_learning_dataset.json`, and
`surgical_learning_dataset2` uses
`modality_surgical_learning_dataset2.json`.

This naming is important because the two UCSD datasets are not schema-identical.
They represent the same dVRK embodiment, but their recorded
`observation.state` layouts differ slightly, so each dataset needs its own
modality mapping to align the raw dataset fields with the common training keys.

| Dataset | Modality File |
|---------|---------------|
| `surgical_learning_dataset` | `open_h/embodiments/ucsd_dvrk/modality_surgical_learning_dataset.json` |
| `surgical_learning_dataset2` | `open_h/embodiments/ucsd_dvrk/modality_surgical_learning_dataset2.json` |

### State / Action Keys

Both modality files expose the same high-level keys:

**State keys:**
- `psm_retraction_pose` (7D: xyz + qw,qx,qy,qz)
- `psm_retraction_gripper` (1D)
- `psm_cutter_pose` (7D: xyz + qw,qx,qy,qz)
- `psm_cutter_gripper` (1D)

**Action keys (same names, REL_XYZ_ROT6D for pose, ABSOLUTE for gripper):**
- `psm_retraction_pose` (7D)
- `psm_retraction_gripper` (1D)
- `psm_cutter_pose` (7D)
- `psm_cutter_gripper` (1D)

**Language key:** `task`

## Utilities

Run `prepare_datasets.sh` separately for each sub-dataset with the matching
modality file (the script copies the modality JSON into `meta/` automatically):

```bash
# surgical_learning_dataset
bash open_h/prepare_datasets.sh \
    --embodiment-tag UCSD_DVRK \
    --modality-json open_h/embodiments/ucsd_dvrk/modality_surgical_learning_dataset.json \
    /path/to/UCSD/surgical_learning_dataset

# surgical_learning_dataset2
bash open_h/prepare_datasets.sh \
    --embodiment-tag UCSD_DVRK \
    --modality-json open_h/embodiments/ucsd_dvrk/modality_surgical_learning_dataset2.json \
    /path/to/UCSD/surgical_learning_dataset2
```
