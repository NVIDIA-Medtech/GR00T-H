# TUM SonATA Ultrasound Dataset

## Overview

SonATA is a robotic sonography dataset from TUM's Computer Aided Medical Procedures (CAMP) Lab. It integrates synchronized ultrasound imaging, external visual data, contact force measurements, robot motion data, and textual instructions collected from abdomen, thyroid, and arm phantoms.

## Embodiment Configuration

| Property | Value |
|----------|-------|
| **Embodiment Tag** | `TUM_SONATA_FRANKA` |
| **Config File** | `open_h/embodiments/tum_sonata_franka/tum_sonata_franka_config.py` |
| **Modality Mapping** | `open_h/embodiments/tum_sonata_franka/modality.json` |

## Subsets

| Subset | Episodes | Frames | Tasks | Status |
|--------|----------|--------|-------|--------|
| SonATA_abdomen | 1,533 | 325,634 | 287 | Ready |
| SonATA_arm | ~1,107 | - | - | Ready |
| SonATA_thyroid | ~780 | - | - | Ready |

## Data Format

Already in **LeRobot v2.1** format with proper metadata.

### Raw Dataset Features

| Feature | Shape | Type | Description |
|---------|-------|------|-------------|
| `action` | 6D | float32 | End-effector pose absolute (x, y, z, roll, pitch, yaw) |
| `observation.state` | 7D | float32 | Joint angles (joint_1 through joint_7) |
| `observation.images.tpv_camera` | 480x640x3 | video | Third-person view camera (AV1 codec) |
| `observation.images.wrist_camera` | 480x640x3 | video | Wrist-mounted camera (AV1 codec) |
| `observation.images.ultrasound` | 480x640x3 | video | Ultrasound image (AV1 codec) |
| `observation.meta.force_torque` | 6D | float32 | Force/torque (fx, fy, fz, tx, ty, tz) |
| `observation.meta.probe_type` | 1 | string | Probe model (linear or convex) |
| `observation.meta.probe_acquisition_param` | 6D | float32 | Ultrasound acquisition parameters |
| `observation.meta.probe_cali_mtx` | 7D | float32 | Probe calibration (xyz + quaternion) |
| `instruction.text` | 1 | string | Natural language instruction |
| `instruction.task` | 1 | string | Task category |
| `instruction.sub_task` | 1 | string | Sub-task description |

### State

- `joint_angles` (7D) -- joint angles from `observation.state[0:7]`, embedded via mean/std
- `force_torque` (6D) -- force/torque from `observation.meta.force_torque[0:6]`, embedded via mean/std
- `eef_pose` (6D) -- EEF pose from `action[0:6]`, pass-through reference for REL_XYZ_ROT6D (not embedded)

### Action

- `eef_pose` (6D) -- absolute EEF pose (xyz + roll/pitch/yaw Euler angles)

Action representation: `REL_XYZ_ROT6D` (Euler angle input -> relative xyz + rot6d output = 9D per timestep).
Action horizon: 50 (~1.67s at 30 Hz). `delta_indices = 1..H`.

Position: relative to reference EEF position (delta xyz).
Rotation: Euler angles (RPY) converted to rot6d relative to reference.

### Cameras

- `tpv_camera` -- third-person view (scene context)
- `wrist_camera` -- wrist-mounted (probe positioning)
- `ultrasound` -- ultrasound image

### Language

- `task` -- mapped from `task_index` via `tasks.jsonl`

### Robot Configuration

| Property | Value |
|----------|-------|
| Robot | Franka Panda |
| FPS | 30 Hz |
| Video Codec | AV1 |
| Resolution | 480x640 |

### Data Splits (SonATA_abdomen)

| Split | Episodes |
|-------|----------|
| Train | 0-1071 |
| Val | 1071-1225 |
| Test | 1225-1533 |

## Task Types

1. **Placement** - Position probe on target anatomy
2. **Scanning** - Sweep probe across anatomical region
3. **Navigation** - Move probe to specific landmarks

## Unique Characteristics

- **Multimodal**: Ultrasound + RGB cameras + force/torque
- **Language-conditioned**: Rich natural language instructions with multiple phrasings per task
- **Calibration data**: Camera and probe calibration matrices included
- **Ultrasound metadata**: Probe parameters (frequency, depth, FOV) per episode

## Example Instructions

```
"Place the probe on the middle of the abdomen."
"Perform a transverse sweep across the aorta and IVC."
"Glide across the abdomen transversely covering the aorta and IVC."
```

## Dataset Preparation

Use the shared preparation script to copy the modality JSON into each dataset's `meta/` folder and generate normalization statistics:

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag TUM_SONATA_FRANKA \
    --modality-json open_h/embodiments/tum_sonata_franka/modality.json \
    /path/to/dataset
```
