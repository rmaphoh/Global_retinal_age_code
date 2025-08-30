
#python EyeQ_process_main.py --data_dir /home/jupyter/github_bags/retinal_age_train/Retinal_age

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --model RETFound_dinov2 \
    --savemodel \
    --global_pool \
    --batch_size 32 \
    --world_size 1 \
    --epochs 5 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 0 \
    --data_path /home/jupyter/github_bags/retinal_age_train/Retinal_age \
    --output_dir /home/jupyter/github_bags/retinal_age_train/Retinal_age/output_dir \
    --input_size 224 \
    --task RETFound_retinal_age \
    --finetune RETFound_dinov2_meh

