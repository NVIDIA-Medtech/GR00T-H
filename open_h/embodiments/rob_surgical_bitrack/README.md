# Rob Surgical (bitrack) Dataset

This directory contains the GR00T N1.6 integration assets for the Rob Surgical
bitrack dataset. The dataset uses LeRobot v2.1 format with a single endoscope
video stream and 3-arm Cartesian kinematics (left, right, aux).

## Key characteristics

- **FPS:** 30 Hz
- **Camera:** `observation.images.endoscope` (720 x 1280, AV1, yuv420p)
- **Language:** `instruction.text_with_tool`
- **Arms:** left, right, aux (3 active in model config; `lap_pose` is defined in `modality.json` for data loading but excluded from `rob_surgical_bitrack_config.py`)
- **State (EEF):** 18D = 3 arms * (xyz + roll + pitch + yaw)
- **Action (EEF):** 18D = 3 arms * (xyz + roll + pitch + yaw)
- **Action horizon:** 50 steps (~1.67s at 30 Hz)
- **Action representation:** REL_XYZ_ROT6D with Euler (RPY) input

## Dataset preparation

### 1. Add tool prompts to language instructions

Before training, run the tool prompt script to generate `instruction.text_with_tool` from per-frame tool metadata. This prefixes each language instruction with the active tool names (e.g., `left tool: grasper. right tool: scissors. aux tool: none.`):

```bash
uv run python open_h/embodiments/rob_surgical_bitrack/utils/rob_surgical_add_tool_prompts.py \
    --dataset-path /path/to/rob_surgical_dataset
```

### 2. Generate normalization statistics
```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag ROB_SURGICAL_BITRACK \
    --modality-json open_h/embodiments/rob_surgical_bitrack/modality.json \
    /path/to/rob_surgical_dataset
```

## Modality mapping

See `modality.json` for exact key mappings:
- `endoscope` -> `observation.images.endoscope`
- `left_pose/right_pose/aux_pose` -> slices of `observation.end_effector_state` (`lap_pose` is loaded via `modality.json` but excluded from model config)
- `action` uses the same slices from `action`
- `annotation.instruction` -> `instruction.text_with_tool` (raw text passthrough)

## Base dataset fields (quick note)

- The merged dataset does **not** include gripper/jaw values; EEF state/action are 18D poses only (3 active arms).
- Tool name strings are present (`observation.meta.left_tool/right_tool/aux_tool`).
- Arm visibility flags existed only in one source dataset and were dropped in the merge.

## Action vs state alignment (Rob Surgical)

- `state` is the current EEF pose at timestep `t` (`delta_indices=[0]`).
- `action` is an **absolute** target pose sequence at timesteps `t..t+49`
  (`delta_indices=list(range(50))`), then converted to REL_XYZ_ROT6D w.r.t.
  the current state during training.
- In the merged dataset, `action[t]` is **not** equal to `state[t]` or `state[t+1]`;
  absolute differences can be large (max diff ~800–1200, mean diff ~6–15).

## Notes on EEF NaNs

> **Warning: Data Quality Note** — Some original episodes contain NaNs in the EEF `l_x` and `r_x` components.
> **A cleaning script was used to impute these values from the corresponding action x-values** so
> that REL_XYZ_ROT6D conversion can proceed without dropping frames.
