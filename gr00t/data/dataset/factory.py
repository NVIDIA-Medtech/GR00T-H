import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.split_utils import load_info_json, resolve_episode_indices
from gr00t.data.stats import check_stats_validity, generate_rel_stats, generate_stats
from gr00t.data.types import ActionRepresentation


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def _get_allowed_episode_indices(
        self,
        dataset_path: str,
        embodiment_tag: str,
        include_splits: list[str] | None,
        exclude_splits: list[str] | None,
    ) -> np.ndarray | None:
        """Resolve allowed episode indices based on include/exclude split settings.

        Args:
            dataset_path: Path to the dataset root directory.
            embodiment_tag: Embodiment tag string for loader fallback.
            include_splits: Optional allowlist of split names.
            exclude_splits: Optional denylist of split names.

        Returns:
            Sorted numpy array of allowed episode indices, or None if no filtering
            is requested.
        """
        if not include_splits and not exclude_splits:
            return None

        info = load_info_json(Path(dataset_path))
        total_episodes = info.get("total_episodes")
        if total_episodes is None:
            total_episodes = self._get_episode_count(dataset_path, embodiment_tag)

        allowed = resolve_episode_indices(
            info,
            include_splits=include_splits,
            exclude_splits=exclude_splits,
            total_episodes=int(total_episodes),
        )
        return allowed

    def _get_episode_count(self, dataset_path: str, embodiment_tag: str) -> int:
        """Get the number of episodes in a dataset without loading full data.

        Creates a lightweight episode loader that only reads metadata,
        avoiding expensive video decoding.

        Args:
            dataset_path: Path to the LeRobot format dataset
            embodiment_tag: Embodiment tag for modality config lookup

        Returns:
            Number of episodes in the dataset
        """
        # Create lightweight loader just to get episode count
        # skip_video=True avoids expensive video initialization
        loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=self.config.data.modality_configs[embodiment_tag],
            video_backend=self.config.data.video_backend,
            skip_video=True,
        )
        return len(loader)

    def _stats_exist(self, dataset_path: str) -> bool:
        """Check whether a valid stats.json already exists for the dataset.

        Uses the same validation logic as gr00t.data.stats.check_stats_validity,
        checking that the file exists and contains all required stat fields for
        every float feature defined in info.json.

        Args:
            dataset_path: Path to the dataset root directory.

        Returns:
            True if stats.json exists and is valid; False otherwise.
        """
        dp = Path(dataset_path)
        info_path = dp / "meta" / "info.json"
        if not info_path.exists():
            return False

        with open(info_path, "r") as f:
            features = json.load(f).get("features", {})

        lowdim_features = [k for k, v in features.items() if "float" in v.get("dtype", "")]
        return check_stats_validity(dp, lowdim_features)

    def _get_relative_action_keys(self, embodiment_tag: str) -> list[str]:
        """Resolve action keys that use RELATIVE representation for an embodiment.

        This inspects the configured modality configs rather than the dataset itself,
        because the action representation is defined by the embodiment configuration.

        Args:
            embodiment_tag: Embodiment tag string for modality config lookup.

        Returns:
            Sorted list of action keys that require RELATIVE stats.
        """
        modality_configs = self.config.data.modality_configs[embodiment_tag]
        action_config = modality_configs.get("action")
        if action_config is None or action_config.action_configs is None:
            return []

        relative_keys = []
        for key, config in zip(action_config.modality_keys, action_config.action_configs):
            if config is not None and config.rep == ActionRepresentation.RELATIVE:
                relative_keys.append(key)

        return sorted(relative_keys)

    def _load_percentile_stats(
        self,
        dataset_path: str,
        consolidated_stats: dict | None,
    ) -> dict | None:
        """Load percentile stats for a dataset from consolidated or per-dataset files.

        The lookup order is controlled by whether a consolidated stats mapping is
        provided. When consolidated stats are available, we only use those to avoid
        mixing sources with potentially different normalization assumptions.

        Args:
            dataset_path: Path to the dataset root directory.
            consolidated_stats: Optional consolidated stats mapping keyed by repo_id.

        Returns:
            Percentile stats dictionary for the dataset, or None if not found.
        """
        if consolidated_stats is not None:
            repo_id = Path(dataset_path).name
            return consolidated_stats.get(repo_id)

        stats_path = Path(dataset_path) / "meta" / "temporal_stats.json"
        if not stats_path.exists():
            return None

        with open(stats_path, "r") as f:
            return json.load(f)

    def _percentile_stats_have_relative_action(
        self,
        dataset_path: str,
        embodiment_tag: str,
        consolidated_stats: dict | None,
    ) -> bool:
        """Check whether percentile stats already include RELATIVE action entries.

        This prevents recomputing relative_stats.json when percentile-based stats
        already contain a "relative_action" section for the relevant action keys.
        The check is intentionally lightweight (key existence only) to avoid heavy
        schema validation during dataset initialization.

        Args:
            dataset_path: Path to the dataset root directory.
            embodiment_tag: Embodiment tag string for modality config lookup.
            consolidated_stats: Optional consolidated stats mapping keyed by repo_id.

        Returns:
            True if percentile stats exist and include relative action stats for all
            RELATIVE action keys; False otherwise.
        """
        relative_action_keys = self._get_relative_action_keys(embodiment_tag)
        if not relative_action_keys:
            # No RELATIVE actions configured; nothing to generate.
            return True

        percentile_stats = self._load_percentile_stats(dataset_path, consolidated_stats)
        if not percentile_stats:
            return False

        relative_stats = percentile_stats.get("relative_action")
        if not isinstance(relative_stats, dict):
            return False

        return all(key in relative_stats for key in relative_action_keys)

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        assert self.config.training.eval_strategy == "no", (
            "Sharded dataset does not support evaluation sets"
        )

        all_datasets = []
        all_weights = []

        consolidated_percentile_stats = None

        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets_for_spec = []

            for dataset_path in dataset_spec.dataset_paths:
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"

                # Generate statistics on rank 0, then barrier for distributed
                allowed_episode_indices = self._get_allowed_episode_indices(
                    dataset_path=dataset_path,
                    embodiment_tag=embodiment_tag,
                    include_splits=dataset_spec.include_splits,
                    exclude_splits=dataset_spec.exclude_splits,
                )
                if allowed_episode_indices is not None:
                    print(
                        f"Dataset {dataset_path}: using {len(allowed_episode_indices)} episodes "
                        f"after split filtering"
                    )

                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        if not self._stats_exist(dataset_path):
                            generate_stats(dataset_path, episode_indices=allowed_episode_indices)
                        if not self._percentile_stats_have_relative_action(
                            dataset_path=dataset_path,
                            embodiment_tag=embodiment_tag,
                            consolidated_stats=consolidated_percentile_stats,
                        ):
                            generate_rel_stats(
                                dataset_path,
                                EmbodimentTag(embodiment_tag),
                                episode_indices=allowed_episode_indices,
                            )
                    torch.distributed.barrier()
                else:
                    if not self._stats_exist(dataset_path):
                        generate_stats(dataset_path, episode_indices=allowed_episode_indices)
                    if not self._percentile_stats_have_relative_action(
                        dataset_path=dataset_path,
                        embodiment_tag=embodiment_tag,
                        consolidated_stats=consolidated_percentile_stats,
                    ):
                        generate_rel_stats(
                            dataset_path,
                            EmbodimentTag(embodiment_tag),
                            episode_indices=allowed_episode_indices,
                        )

                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    video_backend=self.config.data.video_backend,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                    episode_indices=allowed_episode_indices,
                )
                datasets_for_spec.append(dataset)

            # Calculate relative weights for training datasets in this spec
            dataset_lengths = np.array([len(d) for d in datasets_for_spec])
            if dataset_lengths.sum() > 0:
                dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            else:
                dataset_relative_lengths = np.ones(len(datasets_for_spec)) / len(datasets_for_spec)

            for dataset, relative_length in zip(datasets_for_spec, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        return (
            ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            ),
            None,
        )
