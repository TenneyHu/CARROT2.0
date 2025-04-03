#debugging demo "./data/recipe_adaption_demo.txt"

#retrieve only
#python main.py --llm_generate 0 --query_rewrite 0 --reranking 0 --debugging 0 --k_per_querytype 20 --input_file_dir="./data/input.txt"

#reranking
#python main.py --llm_generate 0 --query_rewrite 1 --reranking 1 --debugging 0 --k_per_querytype 21 --final_k 63 --input_file_dir="./data/input.txt"

#diverse reranking
#python main.py --llm_generate 0 --query_rewrite 1 --reranking 1 --reranking_type="MMR" --debugging 0 --k_per_querytype 21 --final_k 21 --reranking_alpha 0.7 --input_file_dir="./data/input.txt" --output_file_dir="res/mmr"

#python main.py --llm_generate 0 --query_rewrite 1 --reranking 1 --reranking_type="vendi" --debugging 0 --k_per_querytype 21 --final_k 21 --reranking_alpha 0.7 --input_file_dir="./data/input.txt" --output_file_dir="res/vendi"

python main.py --input_file_dir="./data/input.txt" --outpu