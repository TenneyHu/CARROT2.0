#!/bin/bash
alphas=(0.2 0.3 0.4 0.5 0.6 0.7 0.8)

# 公共参数
COMMON_ARGS="--llm_generate 0 --query_rewrite 1 --reranking 1 --debugging 0 --k_per_querytype 10 --final_k 10 --input_file_dir=./data/input.txt"

# 遍历 reranking_type
for rerank_type in "MMR" "vendi"
do
    for alpha in "${alphas[@]}"
    do
        OUTPUT_DIR="res/${rerank_type,,}_alpha${alpha}"
        echo "Running: $rerank_type with alpha=$alpha"
        python main.py $COMMON_ARGS --reranking_type="$rerank_type" --reranking_alpha "$alpha" --output_file_dir="$OUTPUT_DIR"
    done
done
