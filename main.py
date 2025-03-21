import argparse
import sys
import logging
from carrot_rag import CarrotRetriever, carrot_query_processing
from carrot_prompt import load_prompt
from carrot_processing import load_data, load_recipe_adaption_query, recipe_split
from llama_index.llms.ollama import Ollama

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        print("Prompt Key: ", k)
        print("Text: ", p.get_template())

def parse_args():
    parser = argparse.ArgumentParser(description='Recipe RAG System')
    parser.add_argument('--emb_model', type=str, default='jinaai/jina-embeddings-v2-base-es',
                        help='Name of the embedding model to use')
    parser.add_argument('--llm', type=str, default='llama3.1',
                        help='Name of the llm to use')
    parser.add_argument('--index_dir', type=str, default='/data1/zjy/spanish_adaption_index/',
                        help='Directory to save/load the index')
    parser.add_argument('--save_index', type=int, default=0,
                        help='switch of saving index to the disk')
    parser.add_argument('--debugging', type=int, default=1,
                        help='display rettrieve logs')
    parser.add_argument('--query_rewrite', type=int, default=1,
                        help='switch of rewriting')
    parser.add_argument('--reranking', type=int, default=1,
                        help='switch of reranking')
    parser.add_argument('--task', type=str, default="recipe adaption",
                        help='The task you want to do')
    parser.add_argument('--input_file_dir', type=str, default="./data/recipe_adaption_demo.txt",
                        help='input file dir')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    titles, full_document_maps = load_data(filter_county="ESP")
    print(f"Loaded {len(titles)} docs")
    
    carrot_retriever = CarrotRetriever(
        emb_model=args.emb_model,
        index_dir=args.index_dir,
        full_document_maps=full_document_maps,
        debugging = args.debugging,
        reranking = args.reranking,
        res_num_per_query=5,
        final_res_num = 5
    )

    #indexing the corpus
    if args.save_index: 
        carrot_retriever.save_index(titles)
    
    carrot_retriever.load_index()
    #user query recipe 
    queries = load_recipe_adaption_query(args.input_file_dir)
    llm = Ollama(model=args.llm, request_timeout=300.0)
    for query in queries:
        title, content = recipe_split(query)
        transformed_queries = carrot_query_processing(title, content, args.debugging, args.query_rewrite)
        
        contexts = carrot_retriever.retrieve(transformed_queries)
        contexts = "\n\n".join([n.node.get_content() for n in contexts])
 
        carrot_prompt = load_prompt(args.task)
        response = llm.complete(
            carrot_prompt.format(context_str=contexts, query_str=query)
        )

        print (str(response))
  
    
    