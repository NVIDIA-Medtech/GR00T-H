# CMR Versius

## Embodiment Configuration

| Property | Value |
|----------|-------|
| **Embodiment Tag** | `cmr_versius` |
| **Projector Index** | 4 |
| **Config File** | `open_h/embodiments/cmr_versius/cmr_versius_config.py` |

This enables combined training with other surgical embodiments, with embodiment-specific learned projections in the model.

---

## Action & State Format

CMR Versius records **hand controller (haptic) poses**, not robot end-effector poses. All kinematics represent the surgeon's controller position in camera frame. Actions and state are reformatted with left arm first, then right arm, pose components grouped together (xyz + rotation + gripper). State uses quaternion (xyzw); actions use 6D rotation for REL_XYZ_ROT6D.

### Action Space (20D)

| Idx | Name | Type | Representation | Description |
|-----|------|------|----------------|-------------|
| **Left Arm (0-9)** |||||
| 0-2 | `xyz_left` | float | REL_XYZ_ROT6D | Translation relative to current hand controller pose |
| 3-8 | `rot6d_left` | float | REL_XYZ_ROT6D | 6D rotation relative to current hand controller pose |
| 9 | `pince_left` | float | ABSOLUTE | Gripper/pince [0-1] (sample-and-hold during clutch) |
| **Right Arm (10-19)** |||||
| 10-12 | `xyz_right` | float | REL_XYZ_ROT6D | Translation relative to current hand controller pose |
| 13-18 | `rot6d_right` | float | REL_XYZ_ROT6D | 6D rotation relative to current hand controller pose |
| 19 | `pince_right` | float | ABSOLUTE | Gripper/pince [0-1] (sample-and-hold during clutch) |

**Pass-Through Keys** (used for processing, removed before model):

| Name | Type | Purpose |
|------|------|---------|
| `hapticengaged_left` | bool[T] | Per-timestep clutch state for left arm |
| `hapticengaged_right` | bool[T] | Per-timestep clutch state for right arm |

### State Space (16D embedded)

| Name | Dims | Description |
|------|------|-------------|
| `left_pose` | 7D | Hand controller xyz + quat_xyzw (from action column) |
| `left_gripper` | 1D | Pince [0-1] |
| `right_pose` | 7D | Hand controller xyz + quat_xyzw (from action column) |
| `right_gripper` | 1D | Pince [0-1] |

**Pass-Through Keys** (used for processing, not sent to model):

| Name | Dims | Purpose |
|------|------|---------|
| `translation_scaling` | 1D | Motion scaling factor for REL_XYZ_ROT6D conversion |
| `rotation_scaling` | 1D | Rotation scaling factor for REL_XYZ_ROT6D conversion |
| `hapticengaged_left` | 1D | Clutch state for left arm (filtering + delta re-integration) |
| `hapticengaged_right` | 1D | Clutch state for right arm (filtering + delta re-integration) |

Instrument type, arm color, and arm-to-haptic linkage are encoded as per-timestep **language prompts** (`instruction.text_with_state`) rather than state embeddings. See [State Prompt Preprocessing](#state-prompt-preprocessing).

### CMR-Specific Details

- **Extra context**: Electrosurgery mode, arm linking, instrument type, arm color — sent via language prompts, not state embedding
- **Clutch handling**: Full clutch-aware processing pipeline (see below)
- **State source**: Extracted from `action` column via `original_key` in modality.json

---

## Clutch-Aware Processing

CMR Versius data presents a unique challenge: surgeons frequently **clutch out** (disengage) during procedures to reposition their hands without moving the robot arms. Computing relative actions naively across these clutch events produces invalid training data.

### The Problem

In teleoperation with clutch:
- **Controller (master):** Moves freely when disengaged
- **Robot (slave):** Holds position when disengaged
- **Data recorded:** Controller position (which diverges from robot during clutch)

Standard REL_XYZ_ROT6D computation (`action[t] - state[ref]`) fails because:
1. **Phantom jumps:** Controller repositioning during clutch appears as large "movements"
2. **Invalid targets:** Model learns to predict controller motion, not robot motion
3. **Gripper drops:** Zeroing gripper during clutch teaches model to "drop the needle"

### Dataset Statistics

Analysis of cholecystectomy data (100 episodes, 289K samples):
- **76.3%** of episodes have clutch transitions
- **Mean 5.7** clutch events per episode
- **13.4%** of samples have engagement changes within 2s action horizon

### The Solution: Multi-Stage Pipeline

#### Stage 1: Load-Time Filtering

Automatically discards samples that cannot produce valid training data:
- `armlinkedtohaptic` changes within action horizon (arm swap mid-sequence)
- Both arms fully disengaged for entire horizon (no valid signal)

#### Stage 2: Engagement-Aware Delta Re-integration

Instead of direct subtraction (`pose[t] - pose[ref]`), we:
1. Compute frame-to-frame deltas
2. Mask deltas where either endpoint is disengaged
3. Re-integrate to get cumulative motion

This correctly handles:
- Reference disengaged → later engaged (no phantom jump)
- Mid-horizon clutch events (disengaged deltas zeroed)
- Repositioning during clutch (not counted as arm motion)

```
Standard:  action[t] = pose[t] - pose[ref]     # WRONG: includes clutch repositioning
Ours:      action[t] = Σ(delta[i] * engaged[i]) for i in ref+1..t
```

#### Stage 3: Sample-and-Hold for Absolute Actions

Different action types require different clutch behavior:

| Action Type | Clutch Behavior | Rationale |
|-------------|-----------------|-----------|
| REL_XYZ_ROT6D (pose) | Zero | No movement = zero delta |
| ABSOLUTE (gripper) | Sample-and-hold | Don't drop the needle |

For grippers with `hold_through_clutch=True`:
- `t > 0`: Hold previous value (`action[t] = action[t-1]`)
- `t = 0`: Fall back to robot state (`action[0] = state[gripper]`)

The t=0 fallback is critical because the controller may have snapped to a different position (e.g., gripper opened) but the robot is still holding (e.g., needle grasped).

### Implementation Files

| File | Component |
|------|-----------|
| `gr00t/data/dataset/sharded_single_step_dataset.py` | Load-time filtering |
| `gr00t/data/state_action/pose.py` | Delta re-integration |
| `gr00t/data/state_action/state_action_processor.py` | Sample-and-hold, action zeroing |
| `open_h/embodiments/cmr_versius/cmr_versius_config.py` | ActionConfig with `hold_through_clutch` |

---

## State Prompt Preprocessing

CMR datasets require a preprocessing step that reads per-frame `observation.state`, extracts instrument type / arm color / arm-to-haptic linkage, and writes an `instruction.text_with_state` column into each parquet in-place. Run this before training:

```bash
uv run python open_h/embodiments/cmr_versius/utils/cmr_add_state_prompts.py          # write in-place
uv run python open_h/embodiments/cmr_versius/utils/cmr_add_state_prompts.py --dry-run # preview only
```

---

## Dataset Preparation

Use the shared `open_h/prepare_datasets.sh` script to copy the modality JSON into each dataset's `meta/` folder and generate normalization statistics (`stats.json` and `temporal_stats.json`).

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag CMR_VERSIUS \
    --modality-json open_h/embodiments/cmr_versius/modality.json \
    /path/to/cholecystectomy_50hz_480p /path/to/hysterectomy_50hz_480p ...
```

---

## Notes on State Extraction

In CMR Versius, the `action` column contains the hand controller pose, which **is** the current state (no separate state column). We use `"original_key": "action"` in modality.json to extract pose state, and `delta_indices=[2, 4, ..., 100]` (shifted by 1*FRAME_STRIDE) because action[0] would be identical to state[0].

---

## File Structure

```
open_h/embodiments/cmr_versius/
├── README.md                           # This file
├── cmr_versius_config.py               # Modality config for GR00T training
├── modality.json                       # Index mappings (uses original_key for runtime extraction)
└── utils/
    ├── cmr_add_state_prompts.py        # Adds per-timestep instruction.text_with_state to parquets
    └── cmr_state_prompt_prefix.py      # Shared helper for constructing state prompt prefixes
```

---
