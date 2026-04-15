#!/bin/bash

OUT_DIR=exp/model_deepconf
MODEL_NP=
DATA_FILE=

GPU_NUM=$(nvidia-smi -L | wc -l)
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=32778

torchrun --nproc_per_node=${GPU_NUM} \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/trainer_deepconf.py \
    --config_path conf/ds_zero3.json \
    --model_name_or_path ${MODEL_NP} \
    --out_dir ${OUT_DIR} \
    --data_file ${DATA_FILE} \
    --use_wandb "true" \
    --deepconf_enabled true \
    --window_size 512 \
    --stride 80 \
    --top_p_keep 0.25 \
    --weight_clip_min 0.5 \
    --weight_clip_max 1.5 \
    --standardize_over "batch" \
    --pause_enabled true \
    --tau_pause_quantile 0.45 \
    --tau_abort_quantile 0.05 \
    --max_pauses 3 \
    --max_think_tokens 64 \
    --recovery_bonus 0.08 \
    --leak_penalty 1.0 


# python src/trainer_deepconf.py \
#     --config_path conf/ds_zero3.json \
#     --model_name_or_path ${MODEL_NP} \
#     --out_dir ${OUT_DIR} \
#     --data_file ${DATA_FILE} \
#     --use_wandb false \
#     --deepconf_enabled true \
#     --window_size 2048 \
#     --stride 256 \
#     --top_p_keep 0.25 \
#     --weight_clip_min 0.5 \
#     --weight_clip_max 1.5 \
#     --standardize_over "batch" \
#     --pause_enabled true \
#     --tau_pause_quantile 0.50 \
#     --tau_abort_quantile 0.10 \
#     --max_pauses 2 \
#     --max_think_tokens 32 \
#     --recovery_bonus 0.05 \
#     --leak_penalty 1.0
