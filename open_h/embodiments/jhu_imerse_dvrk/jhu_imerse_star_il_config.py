"""IMERSE star_IL modality configuration for GR00T N1.6.

This configuration supports the JHU IMERSE star_IL dataset with:
- Single-arm KUKA pose actions (7D: xyz + quat_xyzw)
- KUKA joint positions (7D) + endo360 joint (1D) in state
- Endoscope left + wrist left video streams
- Natural language commands from instruction.text

Action Semantics:
- REL_XYZ_ROT6D conversion uses the action pose at t=0 as the reference.
- The reference pose is injected into state as a pass-through key via modality.json.
- In this dataset, the EEF pose in `action` represents the measured point (not a setpoint),
  so using `action[t]` as the state reference is consistent with `action[t] == state[t]`.
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

imerse_star_il_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "endoscope_left",
            "wrist_left",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "kuka_joint_pos",
            "endo360_joint_pos",
            "kuka_pose",
        ],
        mean_std_embedding_keys=[
            "kuka_joint_pos",
            "endo360_joint_pos",
        ],
        pass_through_keys=[
            "kuka_pose",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(1, ACTION_HORIZON + 1)),
        modality_keys=[
            "kuka_pose",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="kuka_pose",
                normalization_type="percentile",
                input_rotation_format="quat",
                reference_rotation_format="quat",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Register with the JHU IMERSE star_IL embodiment tag.
register_modality_config(
    imerse_star_il_config, embodiment_tag=EmbodimentTag.JHU_IMERSE_DVRK_STAR_IL
)
