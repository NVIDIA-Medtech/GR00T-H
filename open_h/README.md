<div align="center">

# Open-H: Multi-Embodiment Healthcare Robot Training

<img src="../media/open-h-collage.jpg" width="800" alt="Open-H Dataset Collage" style="border: 5px solid #76B900;">

*Screen captures from various datasets included in the Open-H dataset.*

</div>

GR00T-H post-trains GR00T N1.6 on surgical robot data from multiple institutions and robot platforms simultaneously. The core challenge is that each institution records data differently — different robots, coordinate conventions, frame rates, camera setups, and state/action representations. GR00T-H solves this by defining per-embodiment modality configs that convert each dataset into a common representation (REL_XYZ_ROT6D for EEF poses) while preserving robot-specific details like clutch handling and motion scaling.

Each embodiment gets its own projector index in the model, enabling embodiment-specific learned projections while sharing the core vision-language-action backbone.

## Documentation

| Guide | Description |
|-------|-------------|
| [Overview](docs/overview.md) | What's different from core GR00T, auto-registration, quick start workflows |
| [Action Configuration](docs/action_configuration.md) | REL_XYZ_ROT6D, rotation formats, the copy-EEF pattern, adding new embodiments |
| [Data Preparation](docs/data_preparation.md) | Stats pipeline, temporal statistics, troubleshooting |
| [Embodiment Comparison](embodiments/README.md) | All 16 embodiments at a glance — dimensions, cameras, action formats |

For core GR00T concepts (LeRobot format, base ModalityConfig, inference API), see [`getting_started/`](../getting_started/).

## Embodiments

All supported embodiments live under `open_h/embodiments/`. Each subdirectory contains:
- A `*_config.py` that defines the modality configuration and registers it with GR00T
- A `modality.json` that maps raw dataset columns/indices to named keys
- A `README.md` with embodiment-specific details (data format, preparation steps, etc.)

See [`open_h/embodiments/README.md`](embodiments/README.md) for a comparison table of all embodiments — covering dataset type, state/action formats, final action dimensions, number of arms, and number of cameras.

Configs are auto-registered on import: `open_h/embodiments/__init__.py` discovers and executes every `*_config.py` file, which calls `register_modality_config()` to populate the global registry. Both `gr00t/experiment/launch_train.py` and `gr00t/experiment/launch_finetune.py` import `open_h.embodiments`, so built-in Open-H embodiments are available automatically.

For built-in Open-H embodiment tags, do not pass `--modality-config-path` during finetuning or stats generation. None of the embodiments under `open_h/embodiments/` require it. The only time you should pass `--modality-config-path` from an Open-H workflow is when you are using the `NEW_EMBODIMENT` tag for a brand-new embodiment that is not already included in the Open-H registry.

## Dataset Preparation

Before training, each dataset needs normalization statistics computed. This is a prerequisite — training will fail without them.

`prepare_datasets.sh` handles the full preparation pipeline for a set of datasets:

```bash
bash open_h/prepare_datasets.sh \
    --embodiment-tag <EMBODIMENT_TAG> \
    --modality-json <path/to/modality.json> \
    /path/to/dataset_a /path/to/dataset_b ...
```

For each dataset, this script:
1. Copies the `modality.json` into the dataset's `meta/` directory
2. Generates `stats.json` (normalization stats over the raw parquet data) via `gr00t/data/stats.py`
3. Generates `temporal_stats.json` (normalization stats for actions after REL_XYZ_ROT6D conversion, with a temporal dimension for the action chunk) via `launch_finetune.py --calculate-norm-stats`

Each dataset gets its own `temporal_stats.json`, but at training time the stats from all datasets sharing the same embodiment tag are merged (weighted by mix ratio) into a single set of normalization statistics per embodiment.

## Training

The primary training config is [`open_h/gr00t_h_config.yaml`](gr00t_h_config.yaml). It specifies all dataset paths, mix ratios, embodiment tags, model settings, and training hyperparameters.

Before launching, replace every `REPLACE_WITH_OPEN_H_DATA_PATH` entry in that YAML with the absolute path to your local Open-H dataset root.

### Multi-Embodiment Training

The released GR00T-H checkpoint was trained on 4 nodes with 8 GPUs each:

```bash
uv run torchrun --nnodes=4 --nproc_per_node=8 \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    gr00t/experiment/launch_train.py \
    --load-config-path open_h/gr00t_h_config.yaml
```

If using fewer nodes or GPUs, adjust both `global_batch_size` and `num_gpus` in the config accordingly. For example, on a single node with 8 GPUs, set `num_gpus: 8` and `global_batch_size: 256`.

Edit `gr00t_h_config.yaml` to add/remove datasets, adjust mix ratios, or change hyperparameters. Each dataset entry specifies an `embodiment_tag` that maps to its registered modality config.

### Single-Embodiment Finetuning

```bash
uv run torchrun --nproc_per_node=8 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-H \
    --dataset-path /path/to/dataset \
    --embodiment-tag <EMBODIMENT_TAG> \
    --num-gpus 8 \
    --global-batch-size 32 \
    --max-steps 20000 \
    --output-dir /path/to/output
```

Key flags:
- `--embodiment-tag`: Must match the tag registered in the config file (e.g., `CMR_VERSIUS`, `JHU_IMERSE_DVRK`)
- `--modality-config-path`: Leave unset for all built-in Open-H embodiments; use it only when finetuning with `NEW_EMBODIMENT` for a brand-new embodiment config
- `--calculate-norm-stats`: Compute normalization statistics and exit (no training)

## File Structure

```
open_h/
├── README.md                # This file
├── __init__.py              # Package marker
├── gr00t_h_config.yaml      # Multi-embodiment GR00T-H training configuration
├── prepare_datasets.sh      # Dataset preparation script (stats generation)
└── embodiments/             # All embodiment definitions
    ├── __init__.py           # Auto-discovers and registers all *_config.py files
    ├── README.md             # Comparison table of all embodiments
    ├── cmr_versius/
    ├── hamlyn_dvrk/
    ├── jhu_imerse_dvrk/
    ├── jhu_lscr_dvrk/
    ├── moon_maestro/
    ├── obuda_dvrk/
    ├── polyu_sim/
    ├── rob_surgical_bitrack/
    ├── sanoscience_sim/
    ├── stanford_dvrk_real/
    ├── tud_tundra_ur5e/
    ├── tum_sonata_franka/
    ├── turin_mitic_ex_vivo/
    ├── ucb_dvrk/
    ├── ucsd_dvrk/
    └── ustc_torin_tuodao/
```
