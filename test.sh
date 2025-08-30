#!/bin/bash

# ====== path to the local data folder ======
ABSOLUTE_PATH="/home/Retinal_age"

CHECKPOINT="${ABSOLUTE_PATH}/output_dir/checkpoint.pth"   # change if stored elsewhere

# ====== External evaluation ======
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=48798 main_external_evaluation.py \
  --model RETFound_dinov2 \
  --savemodel \
  --eval \
  --global_pool \
  --batch_size 32 \
  --world_size 1 \
  --epochs 1 \
  --blr 5e-3 --layer_decay 0.65 \
  --weight_decay 0.05 --drop_path 0.2 \
  --nb_classes 1 \                         # set to 0 if your code uses 0 for regression
  --data_path "${ABSOLUTE_PATH}" \
  --output_dir "${ABSOLUTE_PATH}/output_dir" \
  --input_size 224 \
  --task external_results_MEH \
  --resume "${CHECKPOINT}"
