#!/bin/bash

if [[ $# -eq 2 ]]; then
    GPU_IDS=$1
    MASTER_PORT=$2
else
    echo "Usage: $0 GPU_IDS (example: 0,1,2,3) MASTER_PORT (example: 29501)"
    exit 1
fi

NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

export PYTHONPATH=$PYTHONPATH:$(pwd)

# CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT tools/train.py \
#     --student-model deit_tiny_patch16_224 \
#     --teacher-model deit_small_distilled_patch16_224 \
#     --dataset stanford_cars \
#     --data-path /root/workspace/tactile_llava/dataset/gpt_instruction/misc/AAAKD/dataset \
#     --finetune \
#     --checkpoint /root/workspace/tactile_llava/dataset/gpt_instruction/misc/AAAKD/checkpoints/baseline-deit-tiny/checkpoint.pth \
#     --epochs 1000 \
#     --batch-size 512 \
#     --lr 5e-4 \
#     --weight-decay 1e-4 \
#     --gpus $GPU_IDS \
#     --distillation-type none \
#     --log-file logs/baseline-deit-tiny-stanford-cars.log \
#     --save-dir checkpoints/baseline-deit-tiny-stanford-cars \
#     --wandb \
#     --wandb-project AAAKD

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT tools/train.py \
    --student-model deit_tiny_patch16_224 \
    --teacher-model deit_small_distilled_patch16_224 \
    --dataset flowers \
    --data-path content/AAAKD/dataset \
    --finetune \
    --checkpoint /content/drive/MyDrive/checkpoint_distkd.best.pth \
    --epochs 500 \
    --batch-size 512 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --gpus $GPU_IDS \
    --distillation-type none \
    --log-file logs/new_distkd-deit-tiny-flowers.log \
    --save-dir checkpoints/new_distkd-deit-tiny-flowers \
    --wandb \
    --wandb-project AAAKD_finetuning

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT tools/train.py \
    --student-model deit_tiny_patch16_224 \
    --teacher-model deit_small_distilled_patch16_224 \
    --dataset caltech256 \
    --data-path /content/AAAKD/dataset \
    --finetune \
    --checkpoint /content/drive/MyDrive/checkpoint_distkd.best.pth \
    --epochs 500 \
    --batch-size 512 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --gpus $GPU_IDS \
    --distillation-type none \
    --log-file logs/distkd-deit-tiny-caltech256.log \
    --save-dir checkpoints/distkd-deit-tiny-caltech256 \
    --wandb \
    --wandb-project AAAKD_finetuning
