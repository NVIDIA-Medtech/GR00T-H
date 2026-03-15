"""
Unified processor for robot state and action data.

Handles:
- State normalization (min/max, mean/std, sin/cos encoding)
- Action normalization
- Absolute <-> Relative <-> REL_XYZ_ROT6D action representation conversion
- Action processing with state dependency
"""

from copy import deepcopy

from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.state_action.pose import (
    EndEffectorPose,
    JointPose,
    apply_motion_scaling_to_rel_xyz_rot6d,
    convert_from_rel_xyz_rot6d,
    convert_to_rel_xyz_rot6d,
    convert_to_rel_xyz_rot6d_with_engagement,
    unapply_motion_scaling_from_rel_xyz_rot6d,
)
from gr00t.data.types import (
    CMR_ENGAGED_LEFT_KEY,
    CMR_ENGAGED_RIGHT_KEY,
    EEF_XYZ_ROT6D_DIM,
    ROT6D_IDENTITY,
    XYZ_DIM,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.data.utils import (
    apply_sin_cos_encoding,
    nested_dict_to_numpy,
    normalize_values_meanstd,
    normalize_values_minmax,
    parse_modality_configs,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)
import numpy as np


class StateActionProcessor:
    """
    Unified processor for robot state and action data.

    Handles:
    - State normalization (min/max, mean/std, sin/cos encoding)
    - Action normalization
    - Absolute <-> Relative <-> REL_XYZ_ROT6D action representation conversion
    - Action processing with state dependency
    - CMR clutch-aware zeroing of disengaged actions
    """

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
    ):
        """
        Initialize unified state and action processor.

        Args:
            modality_configs: Nested dict with structure:
                {embodiment_tag: {modality: ModalityConfig}}
                where modality in ["state", "action"]
                Example: {"gr1": {"state": ModalityConfig(...), "action": ModalityConfig(...)}}
            statistics: Optional nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
                where modality in ["state", "action", "relative_action"]
                and stat_type in ["min", "max", "mean", "std", "q01", "q02", "q98", "q99"]
                Example: {"gr1": {"state": {"left_arm": {"min": [...], "max": [...], ...}}}}
            use_percentiles: Whether to use percentiles (q01/q99) instead of min/max
            clip_outliers: Whether to clip normalized values to [-1, 1]
            apply_sincos_state_encoding: Global flag to enable sin/cos encoding for states
            use_relative_action: Whether to convert actions to relative representation
        """
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # Normalization parameters computed from statistics
        self.norm_params: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
        # Format: norm_params[embodiment_tag][modality][joint_group][stat_type]
        # where stat_type in ["min", "max", "mean", "std", "q01", "q02", "q98", "q99", "dim"]

        if statistics is not None:
            self.set_statistics(statistics)

        # Initialize CMR detection for clutch-aware zeroing
        self._init_cmr_action_indices()

        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def _init_cmr_action_indices(self) -> None:
        """Pre-compute whether each embodiment has CMR clutch engagement keys."""
        self._cmr_action_indices: dict[str, bool] = {}

        for embodiment_tag, modality_configs in self.modality_configs.items():
            state_config = modality_configs.get("state")
            if state_config is None:
                self._cmr_action_indices[embodiment_tag] = False
                continue

            state_keys = state_config.modality_keys or []
            self._cmr_action_indices[embodiment_tag] = (
                CMR_ENGAGED_LEFT_KEY in state_keys and CMR_ENGAGED_RIGHT_KEY in state_keys
            )

    def _zero_disengaged_actions(
        self,
        action: dict[str, np.ndarray],
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """Zero out action targets for disengaged timesteps (CMR clutch-aware zeroing).

        For each timestep where hapticengaged=False for a particular arm, zero out all
        action components (pose, energy) for that arm, however absolute actions with
        hold_through_clutch=True will hold the last engaged value (Like the Gripper).
        This teaches the model to predict "no movement" when the surgeon is clutched out.

        This happens PRE-normalization because zero is meaningful in raw action space.

        Temporal Engagement:
            When hapticengaged_left/right are in the ACTION modality_keys, they provide
            per-timestep engagement data [T, 1]. This enables proper zeroing for mid-horizon
            clutch events. The hapticengaged keys are removed from the action dict after
            zeroing (before normalization).

        Error (No Temporal Engagement):
            If hapticengaged is only in STATE (not action), a ValueError is raised.
            Temporal engagement in the action dict is required for proper clutch handling.

        Args:
            action: Dict mapping joint_group -> raw action values [T, D]
            state: Dict mapping joint_group -> raw state values
            embodiment_tag: Embodiment identifier

        Returns:
            Action dict with disengaged timesteps zeroed out (hapticengaged keys removed)
        """
        # Check if this embodiment has CMR clutch engagement keys
        if not self._cmr_action_indices.get(embodiment_tag, False):
            return action  # Not CMR data, pass through unchanged

        # Use centralized CMR engagement key constants from types.py
        engaged_left_key = CMR_ENGAGED_LEFT_KEY
        engaged_right_key = CMR_ENGAGED_RIGHT_KEY

        # Check for temporal engagement in action dict (preferred)
        has_temporal_engagement = engaged_left_key in action and engaged_right_key in action

        if has_temporal_engagement:
            # Temporal Zeroing/Sample-and-Hold: Per-timestep engagement
            #
            # Behavior determined by action representation and hold_through_clutch flag:
            # - RELATIVE/REL_XYZ_ROT6D actions: always zero (no movement)
            # - ABSOLUTE actions with hold_through_clutch=True: hold last engaged value
            #   (e.g., gripper holding a needle - don't drop it)
            # - ABSOLUTE actions with hold_through_clutch=False: zero (fail-safe)
            #   (e.g., energy button - stop firing for safety)
            #
            # Default is fail-safe (zero) unless explicitly marked to hold.
            engaged_left = action[engaged_left_key].astype(bool).flatten()  # [T]
            engaged_right = action[engaged_right_key].astype(bool).flatten()  # [T]

            # Get modality keys and action configs for checking representation type
            modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
            action_configs = self.modality_configs[embodiment_tag]["action"].action_configs
            action_keys = [
                k for k in modality_keys if k not in [engaged_left_key, engaged_right_key]
            ]

            # Build mapping from key to action config for checking hold_through_clutch
            action_configs_by_key = {}
            if action_configs:
                for k, cfg in zip(modality_keys, action_configs):
                    action_configs_by_key[k] = cfg

            # Process each action key individually to handle sample-and-hold correctly
            for key in action_keys:
                action_data = action[key]  # [T, D]
                T = action_data.shape[0]

                # Determine behavior based on action config
                cfg = action_configs_by_key.get(key)
                is_relative = cfg is not None and cfg.rep in [
                    ActionRepresentation.RELATIVE,
                    ActionRepresentation.REL_XYZ_ROT6D,
                ]

                # For ABSOLUTE actions, check hold_through_clutch flag (defined in ActionConfig with default=False)
                # - True: hold last engaged value (e.g., gripper holding a needle)
                # - False: zero during clutch-out (fail-safe default, e.g., energy button)
                should_hold = cfg is not None and not is_relative and cfg.hold_through_clutch

                # Determine which arm this key belongs to
                is_left = "left_" in key
                is_right = "right_" in key

                for t in range(T):
                    # Get engagement for this arm at this timestep
                    engaged = True  # Default: engaged
                    if is_left:
                        engaged = engaged_left[t]
                    elif is_right:
                        engaged = engaged_right[t]

                    if not engaged:
                        if should_hold:
                            # Sample-and-hold for ABSOLUTE actions with hold_through_clutch=True
                            if t > 0:
                                action_data[t] = action_data[t - 1]  # Hold previous value
                            else:
                                # t=0 fallback - use robot state instead of controller value
                                # When disengaged, controller may have snapped to a different position
                                # (e.g., gripper opened) but robot is still holding (e.g., needle)
                                if cfg.state_key and cfg.state_key in state:
                                    # Use robot state (e.g., 1.0 for closed gripper)
                                    state_val = state[cfg.state_key]
                                    if state_val.ndim > 1:
                                        state_val = state_val.flatten()
                                    # Handle dimension mismatch (state may be different shape)
                                    if state_val.shape[0] >= action_data.shape[1]:
                                        action_data[t] = state_val[: action_data.shape[1]]
                                    else:
                                        action_data[t] = state_val[0]
                                else:
                                    # Error: hold_through_clutch=True requires state_key for t=0 fallback
                                    #
                                    # When t=0 is disengaged and hold_through_clutch=True, we need a fallback
                                    # value from the robot's actual state. Without state_key, we cannot look
                                    # up this value, and using the controller value is risky (it may have
                                    # snapped to a different position, e.g., gripper opened to 0.0).
                                    raise ValueError(
                                        f"Action key '{key}' has hold_through_clutch=True but no state_key defined. "
                                        f"This is required when t=0 is disengaged to provide a fallback value. "
                                        f"Add state_key to the ActionConfig for '{key}'."
                                    )
                        else:
                            # Zero for RELATIVE actions OR ABSOLUTE with hold_through_clutch=False
                            # Use format-aware zeroing for rot6d components
                            # rot6d=[0,0,0,0,0,0] is invalid; identity=[1,0,0,0,1,0] means "no rotation"
                            if cfg is not None and cfg.format == ActionFormat.XYZ_ROT6D:
                                # XYZ_ROT6D format: xyz (3D) + rot6d (6D) = 9D
                                # Zero translation, identity rotation
                                action_data[t, :XYZ_DIM] = 0.0
                                action_data[t, XYZ_DIM:EEF_XYZ_ROT6D_DIM] = ROT6D_IDENTITY
                            elif cfg is not None and cfg.format == ActionFormat.ROT6D:
                                # Pure rotation format: use identity
                                action_data[t] = ROT6D_IDENTITY
                            else:
                                # Default: zero (XYZ, DEFAULT, or other formats)
                                action_data[t] = 0.0

                action[key] = action_data

            # Remove hapticengaged keys from action dict (not needed for model)
            del action[engaged_left_key]
            del action[engaged_right_key]

        # CMR data requires per-timestep engagement for proper clutch handling.
        # Reference-frame-only engagement is insufficient because it cannot detect
        # mid-horizon clutch events, doesn't support sample-and-hold for ABSOLUTE
        # actions, and doesn't use format-aware zeroing (identity rotation for rot6d).
        #
        # Fix: Include hapticengaged_left/right in action.modality_keys and
        # action.pass_through_keys in your modality config.
        elif engaged_left_key in state and engaged_right_key in state:
            raise ValueError(
                f"CMR data detected (hapticengaged_left/right in state) but temporal engagement "
                f"not found in action dict for embodiment '{embodiment_tag}'. "
                f"Add '{engaged_left_key}' and '{engaged_right_key}' to action.modality_keys "
                f"and action.pass_through_keys in your modality config for proper clutch handling."
            )

        return action

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """
        Set dataset statistics for normalization.

        Args:
            statistics: Nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
        """
        for key in statistics:
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(statistics[key])
            else:
                print(f"Embodiment tag {key} already in statistics, skipping updating")
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        """Compute and cache normalization parameters from statistics for all embodiments and modalities.

        Statistics are keyed by embodiment_tag.

        Parameters stored per joint group:
        - min, max: For minmax normalization (state)
        - mean, std: For meanstd normalization
        - q02, q98: For percentile normalization (2nd/98th percentiles)
        - q01, q99: For alternate percentile normalization (1st/99th percentiles)
        - dim: Feature dimension
        """
        for embodiment_tag in self.statistics:
            stats_data = self.statistics[embodiment_tag]

            self.norm_params[embodiment_tag] = {}

            for modality in ["state", "action"]:
                if modality not in stats_data:
                    continue

                self.norm_params[embodiment_tag][modality] = {}

                for joint_group, stats in stats_data[modality].items():
                    # Legacy min/max handling
                    if self.use_percentiles:
                        min_vals = np.array(stats["q01"])
                        max_vals = np.array(stats["q99"])
                    else:
                        min_vals = np.array(stats["min"]) if "min" in stats else None
                        max_vals = np.array(stats["max"]) if "max" in stats else None

                    mean_vals = np.array(stats["mean"])
                    std_vals = np.array(stats["std"])

                    # New percentile parameters (q02/q98) for percentile normalization
                    q02_vals = np.array(stats["q02"]) if "q02" in stats else None
                    q98_vals = np.array(stats["q98"]) if "q98" in stats else None
                    q01_vals = np.array(stats["q01"]) if "q01" in stats else None
                    q99_vals = np.array(stats["q99"]) if "q99" in stats else None

                    # Determine dimension from available stats
                    # For temporal stats, shape is (horizon, dim), so use last axis
                    if q02_vals is not None:
                        dim = q02_vals.shape[-1] if q02_vals.ndim > 1 else q02_vals.shape[0]
                    elif q01_vals is not None:
                        dim = q01_vals.shape[-1] if q01_vals.ndim > 1 else q01_vals.shape[0]
                    elif min_vals is not None:
                        dim = min_vals.shape[-1] if min_vals.ndim > 1 else min_vals.shape[0]
                    else:
                        dim = mean_vals.shape[-1] if mean_vals.ndim > 1 else mean_vals.shape[0]

                    self.norm_params[embodiment_tag][modality][joint_group] = {
                        "dim": np.array(dim),
                        "mean": mean_vals,
                        "std": std_vals,
                    }

                    # Add min/max if available
                    if min_vals is not None:
                        self.norm_params[embodiment_tag][modality][joint_group]["min"] = min_vals
                    if max_vals is not None:
                        self.norm_params[embodiment_tag][modality][joint_group]["max"] = max_vals

                    # Add q02/q98 if available
                    if q02_vals is not None:
                        self.norm_params[embodiment_tag][modality][joint_group]["q02"] = q02_vals
                    if q98_vals is not None:
                        self.norm_params[embodiment_tag][modality][joint_group]["q98"] = q98_vals
                    if q01_vals is not None:
                        self.norm_params[embodiment_tag][modality][joint_group]["q01"] = q01_vals
                    if q99_vals is not None:
                        self.norm_params[embodiment_tag][modality][joint_group]["q99"] = q99_vals

            # Override absolute action stats with relative stats where specified
            if (
                embodiment_tag in self.modality_configs
                and "action" in self.modality_configs[embodiment_tag]
            ):
                modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
                action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

                if action_configs is not None:
                    for key, action_config in zip(modality_keys, action_configs):
                        if (
                            action_config.rep == ActionRepresentation.RELATIVE
                            and self.use_relative_action
                        ):
                            if "relative_action" not in stats_data:
                                raise ValueError(
                                    f"Relative action statistics required for embodiment '{embodiment_tag}' "
                                    f"but 'relative_action' not found in statistics"
                                )
                            if key not in stats_data["relative_action"]:
                                raise ValueError(
                                    f"Relative action statistics required for key '{key}' "
                                    f"in embodiment '{embodiment_tag}' but not found"
                                )
                            action_dim = self.norm_params[embodiment_tag]["action"][key]["dim"]
                            self.norm_params[embodiment_tag]["action"][key] = nested_dict_to_numpy(
                                stats_data["relative_action"][key]
                            )
                            self.norm_params[embodiment_tag]["action"][key]["dim"] = action_dim

                        # For EEF XYZ_ROT6D actions, stats are xyz-only (3D) but model output is 9D
                        # (xyz + rot6d). This dimension is used to SPLIT the concatenated model output,
                        # so it must match the model's internal format, not the final output format.
                        # The rot6d→quat conversion (if needed) happens later in unapply_action.
                        if (
                            action_config.type == ActionType.EEF
                            and action_config.format == ActionFormat.XYZ_ROT6D
                            and key in self.norm_params[embodiment_tag].get("action", {})
                        ):
                            # Model internal format is always XYZ_ROT6D (9D) regardless of output format
                            self.norm_params[embodiment_tag]["action"][key]["dim"] = np.array(
                                EEF_XYZ_ROT6D_DIM
                            )

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Apply state processing (normalization, encoding).

        Args:
            state: Dict mapping joint_group -> raw state values
                Shape per group: (..., D) where D is state dimension
            embodiment_tag: Embodiment identifier (e.g., "gr1") for modality config lookup

        Returns:
            Dict mapping joint_group -> processed state values
                - Sin/cos encoded groups: (..., 2*D)
                - Other groups: (..., D)
        """
        normalized_values = {}
        state = deepcopy(state)  # Avoid modifying input

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        # Get pass-through keys (keys that should not be normalized)
        pass_through_keys = None
        state_config = self.modality_configs[embodiment_tag].get("state")
        if state_config and hasattr(state_config, "pass_through_keys"):
            pass_through_keys = state_config.pass_through_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Strategy 1: Sin/cos encoding (doubles dimension)
            if sin_cos_keys and joint_group in sin_cos_keys:
                normalized_values[joint_group] = apply_sin_cos_encoding(state[joint_group])

            # Strategy 2: Mean/std normalization
            elif (
                hasattr(
                    self.modality_configs[embodiment_tag]["state"],
                    "mean_std_embedding_keys",
                )
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_meanstd(state[joint_group], params)
                normalized_values[joint_group] = normalized

            # Strategy 3: Explicit pass-through for auxiliary keys (e.g., motion scaling factors)
            elif pass_through_keys and joint_group in pass_through_keys:
                normalized_values[joint_group] = state[joint_group]

            # Strategy 4: Min/max normalization to [-1, 1]
            elif joint_group in self.norm_params.get(embodiment_tag, {}).get("state", {}):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                normalized = normalize_values_minmax(state[joint_group], params)

                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)

                normalized_values[joint_group] = normalized

            # Error: key not in any normalization strategy and not in pass_through_keys
            else:
                raise KeyError(
                    f"No normalization stats found for state key '{joint_group}' "
                    f"in embodiment '{embodiment_tag}'. Either add this key to pass_through_keys in the state "
                    f"ModalityConfig, or regenerate statistics to include this key."
                )

        return normalized_values

    def unapply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Reverse state processing (denormalization).

        Args:
            state: Dict mapping joint_group -> processed state values
            embodiment_tag: Embodiment identifier

        Returns:
            Dict mapping joint_group -> raw state values

        Raises:
            ValueError: If attempting to reverse sin/cos encoding (not reversible)
        """
        unnormalized_values = {}

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = None
        if self.apply_sincos_state_encoding:
            state_config = self.modality_configs[embodiment_tag].get("state")
            if state_config and hasattr(state_config, "sin_cos_embedding_keys"):
                sin_cos_keys = state_config.sin_cos_embedding_keys

        # Get pass-through keys (keys that should not be normalized)
        pass_through_keys = None
        state_config = self.modality_configs[embodiment_tag].get("state")
        if state_config and hasattr(state_config, "pass_through_keys"):
            pass_through_keys = state_config.pass_through_keys

        for joint_group in self.modality_configs[embodiment_tag]["state"].modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Sin/cos encoding is not reversible
            if sin_cos_keys and joint_group in sin_cos_keys:
                raise ValueError(
                    f"Cannot unapply sin/cos encoding for joint group '{joint_group}' "
                    f"in embodiment '{embodiment_tag}'. This transformation is not reversible."
                )

            # Reverse mean/std normalization
            elif (
                hasattr(
                    self.modality_configs[embodiment_tag]["state"],
                    "mean_std_embedding_keys",
                )
                and self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
                and joint_group
                in self.modality_configs[embodiment_tag]["state"].mean_std_embedding_keys
            ):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized = unnormalize_values_meanstd(state[joint_group], params)
                unnormalized_values[joint_group] = unnormalized

            # Explicit pass-through for auxiliary keys (e.g., motion scaling factors)
            elif pass_through_keys and joint_group in pass_through_keys:
                unnormalized_values[joint_group] = state[joint_group]

            # Reverse min/max normalization
            elif joint_group in self.norm_params.get(embodiment_tag, {}).get("state", {}):
                params = self.norm_params[embodiment_tag]["state"][joint_group]
                unnormalized_values[joint_group] = unnormalize_values_minmax(
                    state[joint_group], params
                )

            # Error: key not in any normalization strategy and not in pass_through_keys
            else:
                raise KeyError(
                    f"No normalization stats found for state key '{joint_group}' "
                    f"in embodiment '{embodiment_tag}'. Either add this key to pass_through_keys in the state "
                    f"ModalityConfig, or regenerate statistics to include this key."
                )

        return unnormalized_values

    def apply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Apply action processing (absolute->relative conversion, normalization).

        Processing order:
        1. Convert absolute actions to relative (if configured)
        2. Normalize actions

        Args:
            action: Dict mapping joint_group -> raw action values
                Shape per group: (T, D) where T is action horizon, D is action dimension
            embodiment_tag: Embodiment identifier
            state: Optional dict mapping joint_group -> raw state values
                Required if any action group uses ActionRepresentation.RELATIVE
                Shape per group: (T_state, D) where last timestep is used as reference

        Returns:
            Dict mapping joint_group -> processed action values
                Shape per group: (T, D)

        Raises:
            ValueError: If state is None but required for relative action conversion
        """
        action = deepcopy(action)  # Avoid modifying input

        # Step 1: Convert absolute actions to relative (if needed)
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative action processing of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    # Determine which state key to use as reference
                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    # Use last state as reference frame
                    reference_state = state[state_key][-1]

                    # Convert absolute to relative
                    action[key] = self._convert_to_relative_action(
                        action=action[key],
                        reference_state=reference_state,
                        action_type=action_config.type,
                        action_format=action_config.format,
                    )

                elif (
                    action_config.rep == ActionRepresentation.REL_XYZ_ROT6D
                    and self.use_relative_action
                ):
                    # REL_XYZ_ROT6D: translation relative to current EEF, rotation relative to initial
                    if state is None:
                        raise ValueError(
                            f"State dict required for REL_XYZ_ROT6D action processing of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    # Reference EEF pose - single state with delta_indices=[0]
                    # Format depends on reference_rotation_format:
                    #   - "quat": xyz (3) + quat (4) = 7D
                    #   - "rot6d": xyz (3) + rot6d (6) = 9D
                    # State shape is (1, dim), so use [0] to get the single reference
                    eef_pose = state[state_key][0]

                    # Check if per-timestep engagement data is available in action dict (CMR data)
                    # This enables engagement-aware delta re-integration to handle clutch events
                    # Use centralized CMR engagement key constants from types.py
                    if "left" in key:
                        engaged_key = CMR_ENGAGED_LEFT_KEY
                    elif "right" in key:
                        engaged_key = CMR_ENGAGED_RIGHT_KEY
                    else:
                        engaged_key = None
                    engaged = action.get(engaged_key) if engaged_key else None

                    if engaged is not None:
                        # Get reference frame engagement from STATE dict.
                        # State has delta_indices=[0], so it contains the reference frame's engagement.
                        # If ref is disengaged, first delta (action[0] - eef_pose) is invalid.
                        ref_engaged = True  # Default for non-CMR data
                        if engaged_key and engaged_key in state:
                            # State engagement is shape [1, 1] or [1,], extract scalar
                            ref_eng_val = state[engaged_key]
                            if ref_eng_val.ndim > 0:
                                ref_eng_val = ref_eng_val.flatten()[0]
                            ref_engaged = bool(ref_eng_val > 0.5)

                        # Engagement-aware delta re-integration correctly handles:
                        # ref disengaged → later engaged (no phantom jump),
                        # mid-horizon clutch events (disengaged deltas zeroed), and repositioning
                        action[key] = convert_to_rel_xyz_rot6d_with_engagement(
                            action_data=action[key],
                            eef_pose=eef_pose,
                            engaged=engaged.flatten().astype(bool),
                            input_rotation_format=action_config.input_rotation_format,
                            reference_rotation_format=action_config.reference_rotation_format,
                            ref_engaged=ref_engaged,
                            input_quat_order=action_config.input_quat_order,
                            reference_quat_order=action_config.reference_quat_order,
                        )
                    else:
                        # Standard conversion for non-CMR data or when engagement not available
                        # Input: action with xyz + quat (7D) or xyz + rot6d (9D)
                        # Output: xyz_rel + rot6d_rel (9D)
                        action[key] = convert_to_rel_xyz_rot6d(
                            action_data=action[key],
                            eef_pose=eef_pose,
                            input_rotation_format=action_config.input_rotation_format,
                            reference_rotation_format=action_config.reference_rotation_format,
                            input_quat_order=action_config.input_quat_order,
                            reference_quat_order=action_config.reference_quat_order,
                        )

                    # Apply motion scaling if configured (CMR Versius specific)
                    # This converts from hand-controller-space to instrument-space
                    if action_config.translation_scaling_key or action_config.rotation_scaling_key:
                        trans_scale = 1.0
                        rot_scale = 1.0

                        if action_config.translation_scaling_key:
                            if action_config.translation_scaling_key not in state:
                                raise KeyError(
                                    f"Translation scaling key '{action_config.translation_scaling_key}' "
                                    f"not found in state dict for embodiment '{embodiment_tag}'"
                                )
                            trans_scale = float(state[action_config.translation_scaling_key][0])

                        if action_config.rotation_scaling_key:
                            if action_config.rotation_scaling_key not in state:
                                raise KeyError(
                                    f"Rotation scaling key '{action_config.rotation_scaling_key}' "
                                    f"not found in state dict for embodiment '{embodiment_tag}'"
                                )
                            rot_scale = float(state[action_config.rotation_scaling_key][0])

                        action[key] = apply_motion_scaling_to_rel_xyz_rot6d(
                            action[key], trans_scale, rot_scale
                        )

        # Step 1.5: Zero out actions for disengaged timesteps (CMR clutch-aware zeroing)
        # This happens PRE-normalization because zero is meaningful in raw action space
        if state is not None:
            action = self._zero_disengaged_actions(action, state, embodiment_tag)

        # Step 2: Normalize actions
        # Skip pass_through_keys - they're used for processing but not sent to model
        normalized_values = {}
        action_config_obj = self.modality_configs[embodiment_tag]["action"]
        pass_through_keys = set(action_config_obj.pass_through_keys or [])

        for idx, joint_group in enumerate(modality_keys):
            # Skip pass_through_keys - they've been used for processing (e.g., clutch-aware zeroing)
            # and should not be normalized or included in the output
            if joint_group in pass_through_keys:
                continue

            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]

            # For EEF actions with XYZ_ROT6D format, only normalize xyz (rot6d is already bounded)
            action_config = (
                action_configs[idx] if action_configs and idx < len(action_configs) else None
            )
            is_eef_xyz_rot6d = (
                action_config is not None
                and action_config.type == ActionType.EEF
                and action_config.format == ActionFormat.XYZ_ROT6D
            )

            normalization_type = (
                action_config.normalization_type if action_config else "temporal_meanstd"
            )

            if normalization_type == "skip":
                # Pass through without normalization (e.g., for pass-through action keys)
                normalized = action[joint_group]
            elif is_eef_xyz_rot6d:
                # XYZ_ROT6D: normalize xyz with meanstd, pass through rot6d
                action_data = action[joint_group]
                xyz = action_data[..., :3]
                rot6d = action_data[..., 3:]
                normalized_xyz = normalize_values_meanstd(xyz, params)
                normalized = np.concatenate([normalized_xyz, rot6d], axis=-1)
            elif normalization_type == "minmax":
                normalized = normalize_values_minmax(action[joint_group], params)
            else:
                # "temporal_meanstd" and "meanstd" both use meanstd normalization
                # When stats have shape (horizon, dim), normalization is per-timestep
                normalized = normalize_values_meanstd(action[joint_group], params)

            normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Reverse action processing (denormalization, relative->absolute conversion).

        Processing order:
        1. Denormalize actions
        2. Convert relative actions to absolute (if configured)

        Args:
            action: Dict mapping joint_group -> processed action values
                Shape per group: (T, D) or (B, T, D) for batched
            embodiment_tag: Embodiment identifier
            state: Optional dict mapping joint_group -> raw state values
                Required if any action group uses ActionRepresentation.RELATIVE
                Shape per group: (T_state, D) or (B, T_state, D) for batched

        Returns:
            Dict mapping joint_group -> raw absolute action values
                Shape per group: (T, D) or (B, T, D) for batched

        Raises:
            ValueError: If state is None but required for relative->absolute conversion
        """
        # Step 1: Unnormalize actions
        # Skip pass_through_keys - they were used for processing (e.g., clutch-aware zeroing)
        # during apply_action and are not present in the normalized action dict.
        unnormalized_values = {}
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs
        action_config_obj = self.modality_configs[embodiment_tag]["action"]
        pass_through_keys = set(action_config_obj.pass_through_keys or [])

        for idx, joint_group in enumerate(modality_keys):
            # Skip pass_through_keys - they were stripped during apply_action and are not
            # present in the normalized output. Mirrors the skip in apply_action.
            if joint_group in pass_through_keys:
                continue

            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            group_values = action[joint_group]

            # For EEF actions with XYZ_ROT6D format, only unnormalize xyz (rot6d was not normalized)
            action_config = (
                action_configs[idx] if action_configs and idx < len(action_configs) else None
            )
            is_eef_xyz_rot6d = (
                action_config is not None
                and action_config.type == ActionType.EEF
                and action_config.format == ActionFormat.XYZ_ROT6D
            )

            normalization_type = (
                action_config.normalization_type if action_config else "temporal_meanstd"
            )

            if normalization_type == "skip":
                unnormalized = group_values
            elif is_eef_xyz_rot6d:
                # Only unnormalize xyz (first 3 dims), pass through rot6d (last 6 dims)
                xyz = group_values[..., :3]
                rot6d = group_values[..., 3:]
                unnormalized_xyz = unnormalize_values_meanstd(xyz, params)
                unnormalized = np.concatenate([unnormalized_xyz, rot6d], axis=-1)
            elif normalization_type == "minmax":
                unnormalized = unnormalize_values_minmax(group_values, params)
            else:
                # "temporal_meanstd" and "meanstd" both use meanstd
                unnormalized = unnormalize_values_meanstd(group_values, params)

            unnormalized_values[joint_group] = unnormalized

        # Step 2: Convert relative actions to absolute (if needed)

        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                # Skip pass_through_keys - not present in unnormalized_values
                # (they were stripped during apply_action normalization)
                if key in pass_through_keys:
                    continue

                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    if state is None:
                        raise ValueError(
                            f"State dict required for relative->absolute conversion of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    # Determine which state key to use as reference
                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    relative_action = unnormalized_values[key]

                    # Handle batched and unbatched cases
                    is_batched = relative_action.ndim == 3
                    if not is_batched:
                        assert relative_action.ndim == 2
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]
                        relative_action = relative_action[None, :]
                    else:
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]

                    # Convert batched relative actions to absolute
                    absolute_actions = []
                    for s, a in zip(reference_state, relative_action):
                        # Use last timestep of state as reference
                        absolute_action = self._convert_to_absolute_action(
                            action=a,
                            reference_state=s[-1],
                            action_type=action_config.type,
                            action_format=action_config.format,
                        )
                        absolute_actions.append(absolute_action)

                    if is_batched:
                        unnormalized_values[key] = np.stack(absolute_actions, axis=0)
                    else:
                        unnormalized_values[key] = absolute_actions[0]

                elif (
                    action_config.rep == ActionRepresentation.REL_XYZ_ROT6D
                    and self.use_relative_action
                ):
                    # Convert REL_XYZ_ROT6D back to absolute
                    if state is None:
                        raise ValueError(
                            f"State dict required for REL_XYZ_ROT6D->absolute conversion of key '{key}' "
                            f"in embodiment '{embodiment_tag}'"
                        )

                    state_key = action_config.state_key if action_config.state_key else key

                    if state_key not in state:
                        raise KeyError(
                            f"Reference state key '{state_key}' not found in state dict "
                            f"for embodiment '{embodiment_tag}'"
                        )

                    rel_xyz_rot6d_action = unnormalized_values[key]

                    # Handle batched and unbatched cases
                    is_batched = rel_xyz_rot6d_action.ndim == 3
                    if not is_batched:
                        assert rel_xyz_rot6d_action.ndim == 2
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]
                        rel_xyz_rot6d_action = rel_xyz_rot6d_action[None, :]
                    else:
                        reference_state = state[state_key]
                        if reference_state.ndim == 2:
                            reference_state = reference_state[None, :]

                    # Get motion scaling values if configured (CMR Versius specific)
                    has_motion_scaling = (
                        action_config.translation_scaling_key or action_config.rotation_scaling_key
                    )
                    num_samples = reference_state.shape[0]
                    trans_scales = np.ones(num_samples, dtype=np.float32)
                    rot_scales = np.ones(num_samples, dtype=np.float32)

                    if has_motion_scaling:
                        if action_config.translation_scaling_key:
                            trans_scales = self._extract_scaling_values(
                                state,
                                action_config.translation_scaling_key,
                                num_samples,
                                is_batched,
                                embodiment_tag,
                            )
                        if action_config.rotation_scaling_key:
                            rot_scales = self._extract_scaling_values(
                                state,
                                action_config.rotation_scaling_key,
                                num_samples,
                                is_batched,
                                embodiment_tag,
                            )

                    # Convert batched REL_XYZ_ROT6D actions to absolute
                    # State has delta_indices=[0], so s[0] is the single reference state
                    absolute_actions = []
                    for sample_idx, (s, a) in enumerate(zip(reference_state, rel_xyz_rot6d_action)):
                        # Reference EEF pose - single state with delta_indices=[0]
                        eef_pose = s[0]

                        # Unapply motion scaling first (convert instrument-space to hand-controller-space)
                        if has_motion_scaling:
                            a = unapply_motion_scaling_from_rel_xyz_rot6d(
                                a,
                                float(trans_scales[sample_idx]),
                                float(rot_scales[sample_idx]),
                            )

                        absolute_action = convert_from_rel_xyz_rot6d(
                            rel_xyz_rot6d_data=a,
                            eef_pose=eef_pose,
                            # Output in original input format (e.g., quat xyzw) to match GT data
                            output_rotation_format=action_config.input_rotation_format,
                            reference_rotation_format=action_config.reference_rotation_format,
                            output_quat_order=action_config.input_quat_order,
                            reference_quat_order=action_config.reference_quat_order,
                        )
                        absolute_actions.append(absolute_action)

                    if is_batched:
                        unnormalized_values[key] = np.stack(absolute_actions, axis=0)
                    else:
                        unnormalized_values[key] = absolute_actions[0]

        return unnormalized_values

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Apply both state and action processing together.

        Convenience method that processes state and action in one call,
        automatically passing raw state to action processor for relative conversion.

        Args:
            state: Dict mapping joint_group -> raw state values
            action: Dict mapping joint_group -> raw action values
            embodiment_tag: Embodiment identifier

        Returns:
            Tuple of (processed_state, processed_action)
        """
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            processed_action = self.apply_action(action, embodiment_tag, state=state)
        else:
            assert not self.training, "Action is required in training mode"
            processed_action = {}
        return processed_state, processed_action

    def unapply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        raw_state: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Reverse both state and action processing together.

        Args:
            state: Dict mapping joint_group -> processed state values
            action: Dict mapping joint_group -> processed action values
            embodiment_tag: Embodiment identifier
            raw_state: Optional dict of raw states for relative->absolute conversion
                If None, will use unapplied state (but won't work for sin/cos encoded states)

        Returns:
            Tuple of (raw_state, raw_action)
        """
        # Unapply state first
        try:
            unapplied_state = self.unapply_state(state, embodiment_tag)
        except ValueError as e:
            if "sin/cos encoding" in str(e) and raw_state is None:
                raise ValueError(
                    "Cannot unapply sin/cos encoded state. Please provide raw_state parameter."
                ) from e
            raise

        # Use provided raw_state if available, otherwise use unapplied state
        state_for_action = raw_state if raw_state is not None else unapplied_state

        # Unapply action
        unapplied_action = self.unapply_action(action, embodiment_tag, state=state_for_action)

        return unapplied_state, unapplied_action

    def get_state_dim(self, embodiment_tag: str, include_sincos_expansion: bool = False) -> int:
        """
        Get total state dimension after processing.

        Args:
            embodiment_tag: Embodiment identifier
            include_sincos_expansion: If True, accounts for sin/cos encoding doubling dimensions

        Returns:
            Total state dimension across model-input state joint groups.
        """
        total_dim = 0
        state_config = self.modality_configs[embodiment_tag]["state"]
        pass_through_keys = set(state_config.pass_through_keys or [])

        # Get sin/cos embedding keys if enabled
        sin_cos_keys = set()
        if self.apply_sincos_state_encoding and hasattr(state_config, "sin_cos_embedding_keys"):
            sin_cos_keys = set(state_config.sin_cos_embedding_keys)

        for joint_group in state_config.modality_keys:
            if joint_group in pass_through_keys:
                continue
            base_dim = self.norm_params[embodiment_tag]["state"][joint_group]["dim"].item()

            # Sin/cos encoding doubles the dimension
            if include_sincos_expansion and joint_group in sin_cos_keys:
                total_dim += base_dim * 2
            else:
                total_dim += base_dim

        return total_dim

    def get_action_dim(self, embodiment_tag: str) -> int:
        """
        Get total action dimension (excluding pass_through_keys).

        Pass-through keys are used for data processing (e.g., clutch-aware zeroing) but
        are not sent to the model. They are excluded from the dimension calculation.

        Args:
            embodiment_tag: Embodiment identifier

        Returns:
            Total action dimension across all joint groups (excluding pass_through_keys)
        """
        total_dim = 0
        action_config = self.modality_configs[embodiment_tag]["action"]
        pass_through_keys = set(action_config.pass_through_keys or [])

        for joint_group in action_config.modality_keys:
            # Skip pass_through_keys - they're not sent to the model
            if joint_group in pass_through_keys:
                continue
            total_dim += self.norm_params[embodiment_tag]["action"][joint_group]["dim"].item()
        return total_dim

    @staticmethod
    def _extract_scaling_values(
        state: dict[str, np.ndarray],
        scaling_key: str,
        num_samples: int,
        is_batched: bool,
        embodiment_tag: str,
    ) -> np.ndarray:
        """Extract per-sample scaling values from state dict.

        Args:
            state: Dict mapping joint_group -> state values
            scaling_key: Key in state dict for the scaling factor
            num_samples: Number of samples (batch size or 1)
            is_batched: Whether the data is batched
            embodiment_tag: Embodiment identifier (for error messages)

        Returns:
            Array of shape (num_samples,) with per-sample scaling values
        """
        if scaling_key not in state:
            raise KeyError(
                f"Scaling key '{scaling_key}' not found in state dict "
                f"for embodiment '{embodiment_tag}'"
            )
        scale_values = np.asarray(state[scaling_key], dtype=np.float32)
        if is_batched and scale_values.ndim > 0 and scale_values.shape[0] == num_samples:
            return np.array(
                [float(np.asarray(scale_values[i]).reshape(-1)[0]) for i in range(num_samples)],
                dtype=np.float32,
            )
        else:
            return np.full(
                num_samples,
                float(scale_values.reshape(-1)[0]),
                dtype=np.float32,
            )

    def _convert_to_relative_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert absolute action to relative action using reference state."""
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"

        if action_type == ActionType.EEF:
            action_chunking = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)

        elif action_type == ActionType.NON_EEF:
            action_chunking = JointActionChunk([JointPose(m) for m in action])
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        relative_action_chunking = action_chunking.relative_chunking(
            reference_frame=reference_frame
        )
        return relative_action_chunking.to(action_format)

    def _convert_to_absolute_action(
        self,
        action: np.ndarray,
        reference_state: np.ndarray,
        action_type: ActionType,
        action_format: ActionFormat,
    ) -> np.ndarray:
        """Convert relative action to absolute action using reference state."""
        assert action.ndim == 2, f"Expected action shape (T, D), got {action.shape}"
        assert reference_state.ndim == 1, f"Expected state shape (D,), got {reference_state.shape}"
        assert reference_state.shape[0] == action.shape[1], (
            f"State dim {reference_state.shape[0]} != action dim {action.shape[1]}"
        )

        if action_type == ActionType.EEF:
            rel_action = EndEffectorActionChunk.from_array(action, action_format)
            reference_frame = EndEffectorPose.from_action_format(reference_state, action_format)

        elif action_type == ActionType.NON_EEF:
            rel_action = JointActionChunk([JointPose(pose) for pose in action])
            reference_frame = JointPose(reference_state)

        else:
            raise ValueError(f"Unknown ActionType: {action_type}")

        abs_action = rel_action.to_absolute_chunking(reference_frame=reference_frame)
        return abs_action.to(action_format)

    def __str__(self) -> str:
        return f"StateActionProcessor(modality_configs={self.modality_configs}, statistics={self.statistics}, use_percentiles={self.use_percentiles}, clip_outliers={self.clip_outliers}, apply_sincos_state_encoding={self.apply_sincos_state_encoding}, use_relative_action={self.use_relative_action})"
