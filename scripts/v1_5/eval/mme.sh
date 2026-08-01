#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

reduction_ratio=$1
max_num_trunction=$2


CKPT=YOUR_MODEL_PATH 
MODEL=llava-v1.5-7b

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ./playground/data/eval/MME/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --sparse \
    --attn_implementation sdpa \
    --pruned_layer 2 \
    --image_token_start_index 35 \
    --image_token_length 576 \
    --reduction_ratio $reduction_ratio \
    --max_num_trunction $max_num_trunction \
    --pivot_image_token 4 \
    --pivot_text_token 4 


cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $MODEL

cd eval_tool

python calculation.py --results_dir answers/$MODEL
