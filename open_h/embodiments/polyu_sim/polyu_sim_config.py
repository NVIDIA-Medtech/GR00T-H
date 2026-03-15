"""PolyU OpenH_Dataset_full modality configuration for GR00T N1.6.

This configuration supports the PolyU surgical dataset with:
- Joint-angle state (10D) that the policy sees
- Cartesian pose state (7D) used as a pass-through reference for REL_XYZ_ROT6D
- End-effector pose actions (7D: xyz + quat_xyzw) converted to REL_XYZ_ROT6D
- Gripper action (1D) sourced from the action column
- Single endoscope video stream

Data Format:
- State: 17D = psm_joints(10) + psm_cartesian_pose(7: xyz + quat_xyzw)
- Action: 8D = psm_cartesian_pose(7) + psm_gripper(1)

Action Semantics:
- Pose actions are sourced from observation.cartesian_state and aligned to t+1.
- REL_XYZ_ROT6D conversion uses psm_cartesian_pose as the reference state.
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


# Action horizon (how many future action steps to predict).
ACTION_HORIZON = 50

polyu_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "endoscope",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "psm_joints",
            "psm_cartesian_pose",
        ],
        pass_through_keys=[
            "psm_cartesian_pose",
        ],
        mean_std_embedding_keys=[
            "psm_joints",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(1, ACTION_HORIZON + 1)),
        modality_keys=[
            "psm_cartesian_pose",
            "psm_gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="psm_cartesian_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
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

# Register with PolyU simulated surgical tag for Open-H data.
register_modality_config(polyu_config, embodiment_tag=EmbodimentTag.POLYU_SIM)
