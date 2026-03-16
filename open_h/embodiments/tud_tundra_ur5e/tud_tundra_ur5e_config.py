"""
TUD TUNDRA modality configuration for GR00T N1.6 (grasping_retraction only).

This configuration supports the TUD TUNDRA UR5e grasping/retraction dataset with:
- Stereo laparoscope video (left/right)
- Joint-position state embedding (EEF pose available as pass-through reference)
- Absolute EEF pose actions derived from observation.state (t+1..t+H)
- REL_XYZ_ROT6D conversion for EEF pose actions
- Gripper command actions from open_gripper (binary)
- Task strings from tasks.jsonl via task_index

Representation note:
The dataset action column contains delta commands (dx, dy, dz, droll). For
REL_XYZ_ROT6D we instead use the absolute base-frame EEF pose from observation.state
at future timesteps as the action source, following the CMR-style +1 offset.
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
# 30 Hz -> 50 frames ~= 1.7 seconds
ACTION_HORIZON = 50

tud_tundra_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "laparoscope_left",
            # "laparoscope_right", # Mono Only
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "joint_position",
            "eef_pose",
        ],
        mean_std_embedding_keys=[
            "joint_position",
        ],
        pass_through_keys=[
            "eef_pose",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(1, ACTION_HORIZON + 1)),
        modality_keys=[
            "eef_pose",
            "gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.REL_XYZ_ROT6D,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_pose",
                normalization_type="temporal_meanstd",
                input_rotation_format="quat",
                reference_rotation_format="quat",
                reference_quat_order="xyzw",
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

# Register with TUD_TUNDRA_UR5E tag for TUNDRA UR5e surgical assistance data
register_modality_config(tud_tundra_config, embodiment_tag=EmbodimentTag.TUD_TUNDRA_UR5E)
