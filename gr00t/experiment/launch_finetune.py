# Launch finetuning for N1.6 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import open_h.embodiments  # noqa: F401 — registers Open-H embodiment configs
import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def calculate_norm_stats_only(ft_config: FinetuneConfig) -> None:
    """
    Calculate normalization statistics for the dataset and exit without training.

    This function loads the dataset with skip_video=True for fast iteration,
    calculates temporal percentile statistics (q01, q02, q98, q99, mean, std),
    and saves them to the specified output path.

    Uses parallel processing by default for faster computation on large datasets.

    Args:
        ft_config: FinetuneConfig containing dataset path, embodiment tag, and output settings
    """
    import json

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.split_utils import load_info_json, resolve_episode_indices
    from gr00t.data.stats import (
        calculate_temporal_percentile_stats,
        calculate_temporal_percentile_stats_parallel,
    )
    from gr00t.data.utils import to_json_serializable

    # Note: For multi-dataset training with different embodiments, run stats calculation
    # separately for each dataset. Stats are keyed by repo_id, so consolidate them
    # into a single JSON file using --norm-stats-output-path.
    embodiment_tag = ft_config.embodiment_tag.value
    dataset_path = Path(ft_config.dataset_path)

    print(f"Calculating normalization statistics for {dataset_path}")
    print(f"Embodiment: {embodiment_tag}")

    # Get modality configs for this embodiment
    if embodiment_tag not in MODALITY_CONFIGS:
        raise ValueError(
            f"Embodiment '{embodiment_tag}' not found in MODALITY_CONFIGS. "
            f"Available: {list(MODALITY_CONFIGS.keys())}"
        )

    modality_configs = MODALITY_CONFIGS[embodiment_tag]

    episode_indices = None
    if ft_config.include_splits or ft_config.exclude_splits:
        info = load_info_json(dataset_path)
        total_episodes = info.get("total_episodes")
        episode_indices = resolve_episode_indices(
            info,
            include_splits=ft_config.include_splits,
            exclude_splits=ft_config.exclude_splits,
            total_episodes=total_episodes,
        )
        assert episode_indices is not None
        print(f"Using {len(episode_indices)} episodes after split filtering")

    # Calculate temporal percentile statistics with skip_video=True
    # Use parallel version by default (num_workers=None uses CPU count)
    num_workers = ft_config.stats_num_workers
    if num_workers == 1:
        # Single worker - use sequential version
        stats = calculate_temporal_percentile_stats(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            skip_video=True,
            max_episodes=-1,
            episode_indices=episode_indices,
            embodiment_tag=ft_config.embodiment_tag,
        )
    else:
        # Parallel version (default)
        stats = calculate_temporal_percentile_stats_parallel(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            skip_video=True,
            max_episodes=-1,
            num_workers=num_workers,
            episode_indices=episode_indices,
            embodiment_tag=ft_config.embodiment_tag,
        )

    # Determine output path
    if ft_config.norm_stats_output_path:
        output_path = Path(ft_config.norm_stats_output_path)
        # If it's a consolidated file, key by dataset name
        if output_path.suffix == ".json":
            # Check if file exists and load existing stats
            if output_path.exists():
                with open(output_path, "r") as f:
                    all_stats = json.load(f)
            else:
                all_stats = {}

            # Add this dataset's stats keyed by repo_id
            repo_id = dataset_path.name
            all_stats[repo_id] = to_json_serializable(stats)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(all_stats, f, indent=2)
            print(f"Saved statistics to {output_path} (keyed by '{repo_id}')")
        else:
            raise ValueError(f"Output path must be a .json file: {output_path}")
    else:
        # Save to dataset's meta directory
        output_path = dataset_path / "meta" / "temporal_stats.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(to_json_serializable(stats), f, indent=2)
        print(f"Saved statistics to {output_path}")

    print("Statistics calculation complete. Exiting without training.")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    # Handle stats-only calculation mode
    if ft_config.calculate_norm_stats:
        calculate_norm_stats_only(ft_config)
        exit(0)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                        "include_splits": ft_config.include_splits,
                        "exclude_splits": ft_config.exclude_splits,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.state_dropout_prob_per_embodiment = ft_config.state_dropout_prob_per_embodiment
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    # Image size configuration - when set, uses letterbox + resize + crop pipeline
    if ft_config.image_size is not None:
        config.model.image_target_size = ft_config.image_size
        # Use specified crop size or default to target size (no crop augmentation)
        config.model.image_crop_size = ft_config.image_crop_size or ft_config.image_size
        # Use torchvision pipeline which has letterbox padding
        config.model.use_albumentations_transforms = False
        config.model.shortest_image_edge = None
        config.model.crop_fraction = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run(config)
