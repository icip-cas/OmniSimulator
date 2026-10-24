#!/usr/bin/env bash
set -euo pipefail

# test
# python src/generate_cusd_dataset.py \
#     --input_path dataset/example_input.json \
#     --output_file outputs/cusd_sft_dataset.json \
#     --max_samples 50

# all
python src/generate_cusd_dataset.py \
    --input_path dataset/example_input.json \
    --output_file outputs/cusd_sft_dataset.json
