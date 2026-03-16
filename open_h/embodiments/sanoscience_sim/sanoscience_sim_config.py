"""
SanoScience modality configuration for GR00T N1.6.

This configuration supports a simulated surgical robot with 4 instruments:
- REL_XYZ_ROT6D action representation for EEF poses
- Temporal mean-std normalization for actions
- Mean-std normalization for state
- Vision+language only training supported via state dropout
- 1 camera view (color)

Dataset: SanoScience v1.2 merged

Training groups:
  - Expert_demonstrations:                  5,454 episodes,  603,054 frames, 25 fps
  - NonExpert_full_modalities_clean_final:  6,156 episodes,  713,070 frames, 30 fps
  - NonExpert_partial_modalities_clean_final:  666 episodes,   58,752 frames, 30 fps
  - NonExpert_recovery_clean_final:           126 episodes,   20,376 frames, 30 fps
  - NonExpert_stereo_clean_final:           1,602 episodes,  179,640 frames, 30 fps
  Total: 14,004 episodes, 1,574,892 frames

Episode Length Statistics (across all 5 training groups):
- Min length: 16 frames
- Max length: 669 frames
- Mean length: 112.5 frames
- Median length: 102 frames
- Quartiles: Q25=90, Q75=122
- 100% of episodes usable with ACTION_HORIZON=8 (all episodes >= 16 frames)
- 99.7% of episodes usable with ACTION_HORIZON=36 (only 36 dropped)

Action Horizon Selection:
- Using ACTION_HORIZON = 36
- 13,968 / 14,004 episodes usable (99.7%) — 36 episodes dropped (< 36 frames)
- Usable training steps: 1,085,094

Data Format:
- State: 32D total from action.cartesian_absolute, split by instrument:
  - inst_0_pose (7D): xyz + quat_xyzw (indices 0-7)
  - inst_0_gripper (1D): gripper_angle_rad (index 7)
  - inst_1_pose (7D): xyz + quat_xyzw (indices 8-15)
  - inst_1_gripper (1D): gripper_angle_rad (index 15)
  - inst_2_pose (7D): xyz + quat_xyzw (indices 16-23)
  - inst_2_gripper (1D): gripper_angle_rad (index 23)
  - inst_3_pose (7D): xyz + quat_xyzw (indices 24-31)
  - inst_3_gripper (1D): gripper_angle_rad (index 31)

- Action: 32D (same format as state, representing setpoints)
  - Extracted from action.cartesian_absolute column via modality.json original_key

- Additional columns available but NOT used by this config:
  - action (64D): full action vector including handle data
  - action.camera_absolute (32D): camera-frame actions
  - action.joint_positions (48D): joint positions from inverse kinematics
  - observation.state (64D): full state vector
  - observation.state.joint_positions (48D)
  - observation.state.joint_velocities (48D)
  - instruction.text: per-frame language (identical to tasks.jsonl task string)

Final Action Output: 40D = 4 instruments x (9D REL_XYZ_ROT6D pose + 1D gripper)
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
# With horizon=36: 99.7% of episodes usable (13,968/14,004), 1,085,094 training steps
# Only 36 episodes dropped (those with < 36 frames)
ACTION_HORIZON = 36

sanoscience_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "camera_color",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Single reference state for REL_XYZ_ROT6D
        modality_keys=[
            # Instrument 0
            "inst_0_pose",
            "inst_0_gripper",
            # Instrument 1
            "inst_1_pose",
            "inst_1_gripper",
            # Instrument 2
            "inst_2_pose",
            "inst_2_gripper",
            # Instrument 3
            "inst_3_pose",
            "inst_3_gripper",
        ],
        # Mean-std normalization for all state keys
        mean_std_embedding_keys=[
            "inst_0_pose",
            "inst_0_gripper",
            "inst_1_pose",
            "inst_1_gripper",
            "inst_2_pose",
            "inst_2_gripper",
            "inst_3_pose",
            "inst_3_gripper",
        ],
    ),
    "action": ModalityConfig(
        # Start at +1 so index 0 is state reference (CMR-style offset)
        delta_indices=list(range(1, ACTION_HORIZON + 1)),  # [1, 2, 3, ..., 36]
        modality_keys=[
            # Instrument 0
            "inst_0_pose",
            "inst_0_gripper",
            # Instrument 1
            "inst_1_pose",
            "inst_1_gripper",
            # Instrument 2
            "inst_2_pose",
            "inst_2_gripper",
            # Instrument 3
            "inst_3_pose",
            "inst_3_gripper",
        ],
        action_configs=[
            # ===== Instrument 0 =====
            # Pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="inst_0_pose",  # Reference state for relative conversion
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",  # Input actions are xyz + quaternion (xyzw)
                reference_rotation_format="quat",  # Reference state is also xyz + quaternion
            ),
            # Gripper: absolute angle (no rotation conversion needed)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="temporal_meanstd",
            ),
            # ===== Instrument 1 =====
            # Pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="inst_1_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
            # Gripper: absolute angle
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="temporal_meanstd",
            ),
            # ===== Instrument 2 =====
            # Pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="inst_2_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
            # Gripper: absolute angle
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="temporal_meanstd",
            ),
            # ===== Instrument 3 =====
            # Pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="inst_3_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
            # Gripper: absolute angle
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="temporal_meanstd",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],  # Uses tasks.jsonl
    ),
}

# Register with SANOSCIENCE_SIM tag for simulated surgical robot finetuning
register_modality_config(sanoscience_config, embodiment_tag=EmbodimentTag.SANOSCIENCE_SIM)
