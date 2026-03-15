"""
UCSD Surgical Learning modality configuration for GR00T N1.6.

This config targets the UCSD surgical learning datasets:
- surgical_learning_dataset (912 episodes, 30 Hz)
- surgical_learning_dataset2 (200 episodes, 30 Hz)

Key design choices:
- Use absolute EEF pose + gripper in state
- Use REL_XYZ_ROT6D for EEF pose actions and ABSOLUTE for gripper
- Use wxyz quaternion ordering (qw, qx, qy, qz)
- Use `task` language key from tasks.jsonl

Note: The two UCSD datasets have different observation.state layouts. Use the
appropriate modality.json for each dataset:
- open_h/embodiments/ucsd_dvrk/modality_surgical_learning_dataset.json
- open_h/embodiments/ucsd_dvrk/modality_surgical_learning_dataset2.json
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
# 30 FPS -> 50 frames = 1.67 seconds of prediction
ACTION_HORIZON = 50

ucsd_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "camera_left",
            # "camera_right", # Mono Only
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Single reference state for REL_XYZ_ROT6D
        modality_keys=[
            "psm_retraction_pose",
            "psm_retraction_gripper",
            "psm_cutter_pose",
            "psm_cutter_gripper",
        ],
        mean_std_embedding_keys=[
            "psm_retraction_pose",
            "psm_retraction_gripper",
            "psm_cutter_pose",
            "psm_cutter_gripper",
        ],
    ),
    "action": ModalityConfig(
        # Start at 1 (index 0 is state reference), go to ACTION_HORIZON
        delta_indices=list(range(1, ACTION_HORIZON + 1)),
        modality_keys=[
            "psm_retraction_pose",
            "psm_retraction_gripper",
            "psm_cutter_pose",
            "psm_cutter_gripper",
        ],
        action_configs=[
            # Retraction pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm_retraction_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="wxyz",
                reference_rotation_format="quat",
                reference_quat_order="wxyz",
            ),
            # Retraction gripper: absolute jaw value
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                normalization_type="temporal_meanstd",
            ),
            # Cutter pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm_cutter_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                input_quat_order="wxyz",
                reference_rotation_format="quat",
                reference_quat_order="wxyz",
            ),
            # Cutter gripper: absolute jaw value
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

# Register with UCSD_DVRK tag for surgical robot finetuning
register_modality_config(ucsd_config, embodiment_tag=EmbodimentTag.UCSD_DVRK)
