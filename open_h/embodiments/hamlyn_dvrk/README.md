# Hamlyn Centre Surgical Robot Dataset

This document describes the Hamlyn dataset, a collection of surgical robot demonstrations recorded on the da Vinci Research Kit (dVRK) at the Hamlyn Centre for Robotic Surgery at Imperial College London.

## Embodiment Tags

Due to different frame rates across tasks, **two embodiment tags** are provided:

| Embodiment Tag | FPS | Action Horizon | Time Window | Tasks |
|----------------|-----|----------------|-------------|-------|
| `hamlyn_dvrk_15hz` | 15 | 25 steps | 1.67s | 7 tasks (knot_tying, needle_grasp_and_handover, peg_transfer, Suturing-1, Suturing-2, suturing_single_loop_2, tissue_lifting) |
| `hamlyn_dvrk_30hz` | 30 | 50 steps | 1.67s | 2 tasks (suturing_single_loop_1, tissue_retraction) |

## Dataset Overview

| Property | Value |
|----------|-------|
| **Robot Type** | dVRK (da Vinci Research Kit) |
| **Format** | LeRobot v2.1 |
| **Total Episodes (cleaned)** | 1,019 |
| **Total Frames (cleaned)** | 552,753 |
| **Total Size** | ~16 GB (cleaned folders only; duplicates archived separately) |
| **Frame Rate** | 15 fps (7 tasks) / 30 fps (2 tasks) |
| **Quaternion Order** | **wxyz** (scalar-first) |

### Tasks (15 Hz) -- `hamlyn_dvrk_15hz`

- knot_tying
- needle_grasp_and_handover
- peg_transfer
- Suturing-1
- Suturing-2
- suturing_single_loop_2
- tissue_lifting

### Tasks (30 Hz) -- `hamlyn_dvrk_30hz`

- suturing_single_loop_1
- tissue_retraction

### Task Details

**Multi-task datasets** (multiple task_index values):
- `peg_transfer`: 5 task variants (different colored pegs)
- `needle_grasp_and_handover`: 2 task variants (right-hand grasp, or left-grasp-then-transfer)

**Single-task datasets** (task_index=0 only):
- `knot_tying`, `Suturing-1`, `Suturing-2`, `tissue_retraction`

## Data Format

### State Representation (16D)

The state is composed of four modality keys:

| Key | Dim | Description |
|-----|-----|-------------|
| `left_arm_pose` | 7D | Left arm xyz position (3D) + quaternion wxyz (4D) |
| `left_arm_gripper` | 1D | Left arm jaw angle (0 = closed, ~0.5-0.6 = open) |
| `right_arm_pose` | 7D | Right arm xyz position (3D) + quaternion wxyz (4D) |
| `right_arm_gripper` | 1D | Right arm jaw angle |

Raw data source: `observation.state.left_arm_cartesian` (8D) and `observation.state.right_arm_cartesian` (8D), sliced via `modality.json`.

### Action Representation (16D)

Actions use `REL_XYZ_ROT6D` for EEF poses and `ABSOLUTE` for grippers:

| Key | Rep | Description |
|-----|-----|-------------|
| `left_arm_pose` | REL_XYZ_ROT6D | Relative translation + 6D rotation for left arm |
| `left_arm_gripper` | ABSOLUTE | Left arm jaw angle |
| `right_arm_pose` | REL_XYZ_ROT6D | Relative translation + 6D rotation for right arm |
| `right_arm_gripper` | ABSOLUTE | Right arm jaw angle |

Raw data source: `action.cartesian_absolute` (16D), sliced via `modality.json`.

### Video Streams

Three camera views are used for training:

| Modality Key | Raw Key | Description |
|--------------|---------|-------------|
| `endoscope` | `observation.images.color` | Stereo endoscope color image |
| `wrist_left` | `observation.images.wrist_left` | Left wrist camera |
| `wrist_right` | `observation.images.wrist_right` | Right wrist camera |

Depth images (`observation.images.depth`) are available in the dataset but are **not** used in training.

Video codec: AV1 or H264 depending on task.

### Language

| Modality Key | Source |
|--------------|--------|
| `task` | `task_index` (from `tasks.jsonl`) |

### Metadata

Additional per-frame metadata:

| Key | Description |
|-----|-------------|
| `observation.meta.left_arm_tool` | Tool attached to left arm (e.g., "debakey_forceps") |
| `observation.meta.right_arm_tool` | Tool attached to right arm (e.g., "needle_driver") |
| `observation.meta.left_arm_tpv_cali_mtx` | Left arm calibration matrix (7D) |
| `observation.meta.right_arm_tpv_cali_mtx` | Right arm calibration matrix (7D) |

Tool types observed:
- `debakey_forceps` - Grasping forceps
- `needle_driver` - For needle manipulation
- `large_needle_driver` - Larger variant

## Data Splits

Each task has predefined train/val/test splits plus recovery and failure episodes:

| Task | Train | Val | Test | Recovery | Failure |
|------|-------|-----|------|----------|---------|
| knot_tying | 0:50 | 50:57 | 57:72 | 72:73 | 73:77 |
| needle_grasp_and_handover | 0:91 | 91:104 | 104:130 | 130:137 | - |
| peg_transfer | 0:207 | 207:237 | 237:296 | 296:315 | 315:317 |
| Suturing-1 | 0:75 | 75:86 | 86:107 | 107:148 | 148:180 |
| Suturing-2 | 0:71 | 71:81 | 81:102 | 102:142 | 142:186 |
| tissue_lifting | 0:51 | 51:58 | 58:73 | - | 73:75 |
| tissue_retraction | 0:50 | 50:57 | 57:71 | 71:73 | 73:75 |

**Note**: `suturing_single_loop_1` is a 30Hz task; `suturing_single_loop_2` and `tissue_lifting` are 15Hz tasks.

## Hamlyn-Specific Details

- **Quaternion order**: wxyz (scalar-first); handled via `input_quat_order`/`reference_quat_order` in config
- **Video naming**: `observation.images.color` (endoscope), `wrist_left`/`wrist_right` (wrist cameras)
- **Extra data**: Tool metadata and depth images available (depth not used in training)

## Directory Structure

```
Hamlyn/
├── knot_tying/
│   ├── data/chunk-000/episode_*.parquet
│   ├── videos/chunk-000/
│   │   ├── observation.images.color/episode_*.mp4
│   │   ├── observation.images.depth/episode_*.mp4
│   │   ├── observation.images.wrist_left/episode_*.mp4
│   │   └── observation.images.wrist_right/episode_*.mp4
│   └── meta/
│       ├── info.json
│       └── tasks.jsonl
├── needle_grasp_and_handover/
├── peg_transfer/
├── Suturing-1/
├── Suturing-2/
├── tissue_retraction/
├── tissue_lifting/
├── suturing_single_loop_1/
└── suturing_single_loop_2/
```

## Dataset Preparation

Use the shared preparation script to copy the modality JSON into each dataset's `meta/` folder and generate normalization statistics:

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag hamlyn_dvrk_30hz \
    --modality-json open_h/embodiments/hamlyn_dvrk/modality.json \
    /path/to/dataset
```

For 15 Hz tasks, replace `hamlyn_dvrk_30hz` with `hamlyn_dvrk_15hz`.

## Usage Notes

### Loading with LeRobot

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load a single task
dataset = LeRobotDataset(
    repo_id="hamlyn/tissue_retraction",
    root="${OPEN_H_DATA_PATH}/Hamlyn/tissue_retraction"
)
```

### Embodiment Configuration

Use the Hamlyn-specific embodiment tags to keep frame rates and quaternion ordering correct:
- `hamlyn_dvrk_15hz` for 15Hz tasks (knot_tying, needle_grasp_and_handover, peg_transfer, Suturing-1, Suturing-2, suturing_single_loop_2, tissue_lifting)
- `hamlyn_dvrk_30hz` for 30Hz tasks (suturing_single_loop_1, tissue_retraction)

Depth images are available but intentionally **not** used. The modality config only uses
the color endoscope stream plus left/right wrist cameras.

### Key Considerations for Training

1. **Frame Rate Variation**: Use the matching Hamlyn tag (`hamlyn_dvrk_15hz` or `hamlyn_dvrk_30hz`) to keep the ~1.67s action horizon consistent
2. **Quaternion Order**: Hamlyn uses w,x,y,z and is handled via `input_quat_order`/`reference_quat_order` in the config
3. **Multi-Task Training**: The `task_index` field indicates which sub-task within a dataset
4. **Recovery/Failure Episodes**: May be useful for training robust policies but should be handled carefully

