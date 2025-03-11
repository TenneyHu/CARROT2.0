import argparse
import sys
import logging
from carrot_rag import CarrotQueryEngine, save_index
from carrot_prompt import load_prompt
from carrot_processing import load_data, load_recipe_adaption_query, recipe_split

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        print("Prompt Key: ", k)
        print("Text: ", p.get_template())

def parse_args():
    parser = argparse.ArgumentParser(description='Recipe RAG System')
    parser.add_argument('--emb_model', type=str, default='sentence-transformers/distiluse-base-multilingual-cased-v1',
                        help='Name of the embedding model to use')
    parser.add_argument('--llm', type=str, default='llama3.1',
                        help='Name of the llm to use')
    parser.add_argument('--index_dir', type=str, default='/data1/zjy/recipe_spanish_index/',
                        help='Directory to save/load the index')
    parser.add_argument('--save_index', type=int, default=0,
                        help='switch of saving index to the disk')
    parser.add_argument('--debugging', type=int, default=1,
                        help='display rettrieve logs')
    parser.add_argument('--query_rewrite', type=int, default=1,
                        help='switch of rewriting')
    parser.add_argument('--task', type=str, default="recipe adaption",
                        help='The task you want to do')
    parser.add_argument('--input_file_dir', type=str, default="./data/recipe_adaption_demo.txt",
                        help='input file dir')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    titles, full_document_maps = load_data(filter_county="ESP")
    print(f"Loaded {len(titles)} docs")
    
    #indexing the corpus
    if args.save_index: 
        save_index(titles, args.emb_model, args.index_dir)
    
    #user query recipe 
    queries = load_recipe_adaption_query(args.input_file_dir)

    carrot_query_engine = CarrotQueryEngine(
        emb_model=args.emb_model,
        index_dir=args.index_dir,
        full_document_maps=full_document_maps,
        llm=args.llm,
        switch_query_rewrite = args.query_rewrite,
        debugging = args.debugging,
        k=4
    )
   
    for query in queries:
        title, content = recipe_split(query)
        transformed_queries = carrot_query_engine.query_processing(title, content)
        query_engine = carrot_query_engine.retrieve(transformed_queries)
        
        print (load_prompt(args.task))
        query_engine.update_prompts(load_prompt(args.task))
        
        #display_prompt_dict(query_engine.get_prompts())

        response = query_engine.query(query)
        print("Response: ", response)
    
    