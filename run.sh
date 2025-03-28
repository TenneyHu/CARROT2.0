#retrieve only
#python main.py --llm_generate 0 --query_rewrite 0 --reranking 0 --debugging 0 --k_per_querytype 20 --input_file_dir="./data/input.txt"

#reranking
python main.py --llm_generate 0 --query_rewrite 1 --reranking 1 --debugging 0 --k_per_querytype 21 --final_k 63 --input_file_dir="./data/input.txt"