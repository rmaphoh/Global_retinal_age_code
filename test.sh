#!/bin/bash

python EyeQ_process_main.py

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
    --nb_classes 0 \
    --data_path /app/Retinal_age/ \
    --output_dir /app/Retinal_age/output_dir \
    --input_size 224 \
    --task external_results_MEH \
    --resume ./checkpoint.pth

