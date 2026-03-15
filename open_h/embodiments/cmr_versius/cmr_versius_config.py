"""
CMR Versius modality configuration for GR00T N1.6.

This configuration supports dual-arm surgical robot (Left and Right hand controllers) with:
- REL_XYZ_ROT6D action representation for EEF poses
- Temporal mean-std normalization for actions
- Clutch-aware filtering and action zeroing
- 1 camera view (endoscope)
- Per-timestep language prompts encoding instrument type, color, and arm linkage

State Fields (2 categories):
  Embedded Keys (mean-std normalized):
    - left_pose (7D): xyz + quat_xyzw from action[0:7]
    - left_gripper (1D): pince from action[10]
    - right_pose (7D): xyz + quat_xyzw from action[13:20]
    - right_gripper (1D): pince from action[23]

  Pass-Through Keys (not embedded, used for processing only):
    - translation_scaling (1D): from observation.state[12] - motion scaling factor
    - rotation_scaling (1D): from observation.state[13] - rotation scaling factor
    - hapticengaged_left (1D): from observation.state[16] - clutch filtering
    - hapticengaged_right (1D): from observation.state[17] - clutch filtering

Language Prompts (per-timestep, from parquet column instruction.text_with_state):
  Previously-embedded state info is now sent through the VLM backbone as language:
    - armlinkedtohaptic_left/right, instrtype_left/right, arm color
    - Format: "arm left: <val>. left instrument: <name> (<color>). arm right: <val>.
               right instrument: <name> (<color>). do a <procedure>"
  See open_h/embodiments/cmr_versius/utils/cmr_add_state_prompts.py for the script that writes these prompts.

Action: Extracted from action array
  - left_pose (7D): xyz + quat_xyzw -> converted to xyz + rot6d (9D)
  - left_gripper (1D): pince
  - right_pose (7D): xyz + quat_xyzw -> converted to xyz + rot6d (9D)
  - right_gripper (1D): pince
REL_XYZ_ROT6D Conversion:
- Pose actions are converted to REL_XYZ_ROT6D: translation and rotation relative to current EEF
- Output: 9D per arm (xyz_rel + rot6d_rel) + 1D gripper = 10D per arm

Final Action Output: 20D = left(10) + right(10)

Clutch-Aware Processing:
- Load-time filtering: Discards samples where armlinkedtohaptic changes within horizon
  or where both arms are fully disengaged
- Action zeroing: Zeros action targets for arms where hapticengaged=False

"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Action horizon (how many future action steps to predict)
ACTION_HORIZON = 50
# Frame stride for downsampling 60Hz -> 30Hz
FRAME_STRIDE = 2

cmr_versius_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "endoscope",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Single reference state for REL_XYZ_ROT6D
        modality_keys=[
            # === Embedded keys (mean-std normalized) ===
            "left_pose",
            "left_gripper",
            "right_pose",
            "right_gripper",
            # === Pass-through keys (never embedded to model) ===
            # Used for action normalization only
            "translation_scaling",
            "rotation_scaling",
            # Used for clutch-aware filtering only
            "hapticengaged_left",
            "hapticengaged_right",
        ],
        # Pass-through keys: NEVER sent to model, only used for data processing
        pass_through_keys=[
            "translation_scaling",  # For action normalization
            "rotation_scaling",  # For action normalization
            "hapticengaged_left",  # For clutch-aware filtering
            "hapticengaged_right",  # For clutch-aware filtering
        ],
        # Mean-std normalization for continuous values (pose, gripper)
        mean_std_embedding_keys=[
            "left_pose",
            "left_gripper",
            "right_pose",
            "right_gripper",
        ],
    ),
    "action": ModalityConfig(
        # [2, 4, 6, ..., 100] - stride=2 for 30Hz effective rate, starts at 2 to skip current state
        delta_indices=list(range(FRAME_STRIDE, ACTION_HORIZON * FRAME_STRIDE + 1, FRAME_STRIDE)),
        modality_keys=[
            "left_pose",
            "left_gripper",
            "right_pose",
            "right_gripper",
            # Haptic engagement for per-timestep action zeroing (removed before normalization)
            "hapticengaged_left",
            "hapticengaged_right",
        ],
        # Pass-through: hapticengaged used for action zeroing, then removed before model
        pass_through_keys=[
            "hapticengaged_left",
            "hapticengaged_right",
        ],
        action_configs=[
            # Left pose: REL_XYZ_ROT6D EEF action with motion scaling
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="left_pose",  # Reference state for relative conversion
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",  # Input actions are xyz + quaternion
                reference_rotation_format="quat",  # Reference state is also xyz + quaternion
                translation_scaling_key="translation_scaling",  # CMR motion scaling
                rotation_scaling_key="rotation_scaling",
            ),
            # Left gripper: absolute pince value (hold through clutch - don't drop needle)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_gripper",  # Enables hold if clutch disengaged at t=1
                normalization_type="temporal_meanstd",
                hold_through_clutch=True,  # Gripper should hold position during clutch-out
            ),
            # Right pose: REL_XYZ_ROT6D EEF action with motion scaling
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="right_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
                translation_scaling_key="translation_scaling",  # CMR motion scaling
                rotation_scaling_key="rotation_scaling",
            ),
            # Right gripper: absolute pince value (hold through clutch - don't drop needle)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_gripper",  # For t=0 fallback when disengaged (Bug 6 fix)
                normalization_type="temporal_meanstd",
                hold_through_clutch=True,  # Gripper should hold position during clutch-out
            ),
            # Haptic engaged left: pass-through for action zeroing (removed before normalization)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="skip",
            ),
            # Haptic engaged right: pass-through for action zeroing (removed before normalization)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="skip",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Register with CMR_VERSIUS tag for surgical robot finetuning
register_modality_config(cmr_versius_config, embodiment_tag=EmbodimentTag.CMR_VERSIUS)
