"""
Rob Surgical (bitrack) modality configuration for GR00T N1.6.

This configuration supports the Rob Surgical dataset with:
- Single endoscope video stream
- 3-arm Cartesian EEF state (left, right, aux)
- 3-arm absolute EEF action targets (xyz + roll/pitch/yaw)
- REL_XYZ_ROT6D action conversion using Euler angles

Data representation:
- State: 18D = 3 * (xyz + roll + pitch + yaw)
- Action: 18D = 3 * (xyz + roll + pitch + yaw)
- Rotation format: Euler RPY (radians), converted to rot6d internally
- Note: lap_pose is defined in modality.json for data loading but is
  excluded from this model config (commented out below).

Notes:
- EEF x-value NaNs were imputed during the dataset merge step by copying
  the action x-values for the affected arm. This config assumes merged data.
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
# 30 Hz -> 50 frames ~= 1.67 seconds
ACTION_HORIZON = 50

rob_surgical_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "endoscope",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],  # Single reference state for REL_XYZ_ROT6D conversion
        modality_keys=[
            "left_pose",
            "right_pose",
            # "lap_pose",
            "aux_pose",
        ],
        mean_std_embedding_keys=[
            "left_pose",
            "right_pose",
            # "lap_pose",
            "aux_pose",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "left_pose",
            "right_pose",
            # "lap_pose",
            "aux_pose",
        ],
        action_configs=[
            # Left arm pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="left_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
            # Right arm pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="right_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
            # Lap arm pose: REL_XYZ_ROT6D EEF action
            # ActionConfig(
            #     rep=ActionRepresentation.REL_XYZ_ROT6D,
            #     type=ActionType.EEF,
            #     format=ActionFormat.XYZ_ROT6D,
            #     state_key="lap_pose",
            #     normalization_type="percentile",
            #     input_rotation_format="euler",
            #     reference_rotation_format="euler",
            # ),
            # Aux arm pose: REL_XYZ_ROT6D EEF action
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="aux_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.instruction"],  # instruction.text (raw strings)
    ),
}

# Register with ROB_SURGICAL_BITRACK tag for Rob Surgical dataset integration
register_modality_config(rob_surgical_config, embodiment_tag=EmbodimentTag.ROB_SURGICAL_BITRACK)
