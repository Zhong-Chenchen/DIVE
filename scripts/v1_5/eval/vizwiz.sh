#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

CKPT=YOUR_MODEL_PATH
MODEL=llava-v1.5-7b

reduction_ratio=$1
max_num_trunction=$2


python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/$MODEL.jsonl \
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

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$MODEL.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$MODEL.json
