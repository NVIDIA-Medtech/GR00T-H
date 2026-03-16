"""
dVRK UCBerkeley modality configuration for GR00T N1.6.

This configuration supports the UCBerkeley debridement dataset with:
- Cartesian EEF pose actions (16D) using REL_XYZ_ROT6D
- Absolute gripper actions per arm
- Joint-angle state channels for normalization and dropout control
- 2 camera views (left, right stereo) instead of 4

Data Format:
- State: cartesian state (xyz + quat + jaw) plus joint angles, split by arm:
  - psm1_pose (7D): [x, y, z, qx, qy, qz, qw]
  - psm1_gripper (1D): jaw
  - psm2_pose (7D): [x, y, z, qx, qy, qz, qw]
  - psm2_gripper (1D): jaw
- Joint angles:
  - psm1_joints (7D)
  - psm2_joints (7D)
- Action: 16D cartesian setpoints (same format as state)

REL_XYZ_ROT6D Conversion:
- Pose actions are converted to relative translation/rotation from the current state.
- action[t] corresponds to state[t+1] (setpoint for the next timestep), so relative
  conversion uses state[t] as the reference frame for the action horizon.

Dataset: UCBerkeley debridement (589 episodes, 221,950 frames)
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
# Matches existing dVRK config for consistency
ACTION_HORIZON = 50

dvrk_ucb_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "camera_left",
            # "camera_right", # Mono Only
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Single reference state at timestep t
        modality_keys=[
            "psm1_joints",
            "psm1_gripper",
            "psm2_joints",
            "psm2_gripper",
            "psm1_pose",
            "psm2_pose",
        ],
        mean_std_embedding_keys=[
            "psm1_joints",
            "psm1_gripper",
            "psm2_joints",
            "psm2_gripper",
        ],
        # Pass-through cartesian pose/gripper state for REL_XYZ_ROT6D reference
        pass_through_keys=[
            "psm1_pose",
            "psm2_pose",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),  # [0, 1, 2, ..., 49] consecutive
        modality_keys=[
            "psm1_pose",
            "psm1_gripper",
            "psm2_pose",
            "psm2_gripper",
        ],
        action_configs=[
            # PSM1 pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm1_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="xyzw",
                reference_rotation_format="quat",
                reference_quat_order="xyzw",
            ),
            # PSM1 gripper: absolute jaw angle
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,
                normalization_type="temporal_meanstd",
            ),
            # PSM2 pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm2_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="xyzw",
                reference_rotation_format="quat",
                reference_quat_order="xyzw",
            ),
            # PSM2 gripper: absolute jaw angle
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
        modality_keys=["task"],  # Uses tasks.jsonl: "surgical_debridement"
    ),
}

# Register with UCB_DVRK tag for UCBerkeley cartesian EEF surgical robot data
register_modality_config(dvrk_ucb_config, embodiment_tag=EmbodimentTag.UCB_DVRK)
