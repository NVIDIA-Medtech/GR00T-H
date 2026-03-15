"""
Hamlyn Centre Surgical Robot Dataset Embodiment Configurations.

This module defines modality configurations for the Hamlyn dataset, which contains
surgical robot demonstrations recorded on the dVRK (da Vinci Research Kit) at
Imperial College London.

Two configurations are provided:
1. `hamlyn_15hz_config`: For 7 tasks recorded at 15Hz (knot_tying, needle_grasp_and_handover,
   peg_transfer, Suturing-1, Suturing-2, suturing_single_loop_2, tissue_lifting)
2. `hamlyn_30hz_config`: For 2 tasks recorded at 30Hz (suturing_single_loop_1, tissue_retraction)

Key differences from JHU dVRK dataset:
- Quaternion ordering: wxyz (scalar-first) vs xyzw (scalar-last)
- Video keys: observation.images.color vs observation.images.endoscope.left
Data format:
- State: 16D Cartesian EEF (left arm 8D + right arm 8D)
  - Per arm: xyz (3D) + quaternion wxyz (4D) + gripper (1D)
- Action: 16D Cartesian absolute (same format as state)
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


# =============================================================================
# 15Hz Configuration (7 tasks)
# =============================================================================
# Tasks: knot_tying, needle_grasp_and_handover, peg_transfer, Suturing-1,
#        Suturing-2, suturing_single_loop_2, tissue_lifting
#
# Action horizon: 25 steps = 1.67 seconds at 15Hz
# This keeps ~85% of samples usable for episodes with 100+ frames.

ACTION_HORIZON_15HZ = 25

hamlyn_15hz_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["endoscope", "wrist_left", "wrist_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm_pose",  # 7D: xyz + quat (wxyz order)
            "left_arm_gripper",  # 1D: jaw angle
            "right_arm_pose",  # 7D: xyz + quat (wxyz order)
            "right_arm_gripper",  # 1D: jaw angle
        ],
        mean_std_embedding_keys=[
            "left_arm_pose",
            "left_arm_gripper",
            "right_arm_pose",
            "right_arm_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON_15HZ)),
        modality_keys=[
            "left_arm_pose",
            "left_arm_gripper",
            "right_arm_pose",
            "right_arm_gripper",
        ],
        action_configs=[
            # Left arm pose: REL_XYZ_ROT6D with wxyz quaternion order
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="left_arm_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="wxyz",  # Hamlyn uses wxyz (scalar-first)
                reference_rotation_format="quat",
                reference_quat_order="wxyz",
            ),
            # Left arm gripper: ABSOLUTE (jaw angle doesn't need relative conversion)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                normalization_type="temporal_meanstd",
            ),
            # Right arm pose: REL_XYZ_ROT6D with wxyz quaternion order
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="right_arm_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="wxyz",
                reference_rotation_format="quat",
                reference_quat_order="wxyz",
            ),
            # Right arm gripper: ABSOLUTE
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                normalization_type="temporal_meanstd",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],  # Uses tasks.jsonl
    ),
}


# =============================================================================
# 30Hz Configuration (2 tasks)
# =============================================================================
# Tasks: suturing_single_loop_1, tissue_retraction
#
# Action horizon: 50 steps = 1.67 seconds at 30Hz (same time window as 15Hz config)
# This keeps ~85% of samples usable for episodes with 100+ frames.

ACTION_HORIZON_30HZ = 50

hamlyn_30hz_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["endoscope", "wrist_left", "wrist_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm_pose",
            "left_arm_gripper",
            "right_arm_pose",
            "right_arm_gripper",
        ],
        mean_std_embedding_keys=[
            "left_arm_pose",
            "left_arm_gripper",
            "right_arm_pose",
            "right_arm_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON_30HZ)),
        modality_keys=[
            "left_arm_pose",
            "left_arm_gripper",
            "right_arm_pose",
            "right_arm_gripper",
        ],
        action_configs=[
            # Left arm pose: REL_XYZ_ROT6D with wxyz quaternion order
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="left_arm_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="wxyz",
                reference_rotation_format="quat",
                reference_quat_order="wxyz",
            ),
            # Left arm gripper: ABSOLUTE
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                normalization_type="temporal_meanstd",
            ),
            # Right arm pose: REL_XYZ_ROT6D with wxyz quaternion order
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="right_arm_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="wxyz",
                reference_rotation_format="quat",
                reference_quat_order="wxyz",
            ),
            # Right arm gripper: ABSOLUTE
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                normalization_type="temporal_meanstd",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}


# Register both configurations
register_modality_config(hamlyn_15hz_config, embodiment_tag=EmbodimentTag.HAMLYN_DVRK_15HZ)
register_modality_config(hamlyn_30hz_config, embodiment_tag=EmbodimentTag.HAMLYN_DVRK_30HZ)
