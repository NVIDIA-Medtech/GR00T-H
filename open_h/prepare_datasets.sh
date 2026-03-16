#!/bin/bash
# Prepare one or more LeRobot datasets for GR00T by generating normalization stats.
#
# This script will:
# 1) Copy a provided modality JSON into each dataset's meta/ folder
# 2) Generate stats.json via gr00t/data/stats.py
# 3) Generate temporal_stats.json (norm stats for REL_XYZ_ROT6D actions over the action chunk)
#
# Run from repo root:
#   bash open_h/prepare_datasets.sh \
#     --embodiment-tag <EMBODIMENT_TAG> \
#     --modality-json <path/to/modality.json> \
#     /path/to/dataset_a /path/to/dataset_b

set -euo pipefail

print_usage() {
    cat <<'EOF'
Usage:
  bash open_h/prepare_datasets.sh \
    --embodiment-tag <EMBODIMENT_TAG> \
    --modality-json <path/to/modality.json> \
    <DATASET_PATH> [<DATASET_PATH> ...]

Required arguments:
  --embodiment-tag   Embodiment tag to pass to stats and finetune scripts
  --modality-json    Path to modality JSON copied into each dataset meta/ folder
  DATASET_PATH       One or more dataset paths to process
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

if [[ ! -f "gr00t/data/stats.py" ]]; then
    die "Run this script from the repo root."
fi

EMBODIMENT_TAG=""
MODALITY_FILE=""
DATASET_PATHS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --embodiment-tag)
            [[ $# -ge 2 ]] || die "Missing value for --embodiment-tag"
            EMBODIMENT_TAG="$2"
            shift 2
            ;;
        --modality-json)
            [[ $# -ge 2 ]] || die "Missing value for --modality-json"
            MODALITY_FILE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                DATASET_PATHS+=("$1")
                shift
            done
            ;;
        -*)
            die "Unknown option: $1"
            ;;
        *)
            DATASET_PATHS+=("$1")
            shift
            ;;
    esac
done

[[ -n "${EMBODIMENT_TAG}" ]] || die "Missing required --embodiment-tag"
[[ -n "${MODALITY_FILE}" ]] || die "Missing required --modality-json"
[[ ${#DATASET_PATHS[@]} -gt 0 ]] || die "Provide at least one DATASET_PATH"
[[ -f "${MODALITY_FILE}" ]] || die "Modality file does not exist: ${MODALITY_FILE}"

for dataset_path in "${DATASET_PATHS[@]}"; do
    echo "=== Processing ${dataset_path} ==="

    [[ -d "${dataset_path}" ]] || die "Dataset path does not exist: ${dataset_path}"

    if [[ ! -d "${dataset_path}/meta" ]]; then
        echo "Creating meta directory: ${dataset_path}/meta"
        mkdir -p "${dataset_path}/meta"
    fi

    echo "Copying ${MODALITY_FILE} to ${dataset_path}/meta/modality.json"
    cp "${MODALITY_FILE}" "${dataset_path}/meta/modality.json"

    echo "Generating stats..."
    uv run python gr00t/data/stats.py \
        --dataset-path "${dataset_path}" \
        --embodiment-tag "${EMBODIMENT_TAG}"

    echo "Generating temporal stats..."
    uv run python gr00t/experiment/launch_finetune.py \
        --base-model-path nvidia/GR00T-N1.6-3B \
        --dataset-path "${dataset_path}" \
        --embodiment-tag "${EMBODIMENT_TAG}" \
        --calculate-norm-stats
done
