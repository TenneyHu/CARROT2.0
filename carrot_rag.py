
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import ollama
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from vendi_score import vendi

def norm(score, min_score=-10.0, max_score =5.0):
    return (score - min_score) / (max_score - min_score)

def recipe_title_generate(recipe):
    response = ollama.chat(model="llama3.1", messages=[
        {"role": "system", "content": "Here is a recipe without title; please create a short Spanish title for the recipe."},
        {"role": "user", "content": f"The recipe is {recipe}"},
        {"role": "user", "content": f" Please only output the recipe title, Do not use quotation marks or include any explanations or additional content."}
    ])
    title = response["message"]["content"]
    return title 

def recipe_title_rewrite(ori_title):
    response = ollama.chat(model="llama3.1", messages=[
        {"role": "system", "content": "Please rewrite the title of this recipe to align with Spanish recipe naming conventions and dietary habits."},
        {"role": "user", "content": f"The recipe is {ori_title}"},
        {"role": "user", "content": f" Please only output the recipe title, Do not use quotation marks or include any explanations or additional content."}
    ])
    title = response["message"]["content"]
    return title

def carrot_query_processing(ori_title, ori_content, debugging, query_rewrite):
        if query_rewrite:
            generated_query = recipe_title_generate(ori_content)
            rewrited_query = recipe_title_rewrite(ori_title)
            res = [ori_title, generated_query, rewrited_query]
        else:
            res = [ori_title]
        
        if debugging and query_rewrite:
            print ("---Query Rewritting---")
            print("Original query: ", ori_title)
            print("Generated query: ", generated_query) 
            print("Rewritten query: ", rewrited_query)
        return res

class RecipeInfo:
    def __init__(self, title, ingredients, steps, is_original_query,
                 position, score):
        self.title = title.replace(", ", ",")  
        self.ingredients = ingredients  
        self.steps = steps  
        self.rerank_text = "Nombre:\t" + self.title + "\t" + "Ingredientes:\t" + self.ingredients
        self.full_text = "Nombre:\t" + self.title + "\t" + "Ingredientes:\t" + self.ingredients + "\t" + "Pasos:\t" + self.steps
        self.from_original_query = is_original_query  
        self.retrieve_position = position
        self.rerank_position = 0  
        self.score = score  
        self.rerank_score = 0.0
        self.diversity_rerank_nth = 99999

    def __repr__(self):
        return (f"title={self.title!r}, from_original_query={self.from_original_query}, "
                f"retrieve_position={self.retrieve_position}, rerank_position={self.rerank_position}, "
                f"score={self.score}, rerank_score={self.rerank_score}, mmr={self.diversity_rerank_nth}, ingredients={self.ingredients!r}")
    
class CarrotRetriever():
    def __init__(self, emb_model, index_dir, full_document_maps, debugging = 0, 
                    reranking = 0, reranking_type = "relevance", reranking_alpha = 0.7, res_num_per_query = 10,  final_res_num = 5):
        self.emb_model = HuggingFaceEmbedding(model_name = emb_model, trust_remote_code = True)
        self.full_document_maps = full_document_maps
        self.index_dir = index_dir
        self.debugging = debugging
        self.index = None
        self.has_reranking = reranking
        self.reranking_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3').to('cuda')
        self.reranking_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        self.reranking_type = reranking_type
        self.reranking_alpha = reranking_alpha
        self.res_num_per_query = res_num_per_query
        self.final_res_num = final_res_num

    def save_index(self, docs):
        docs = [Document(text=doc) for doc in docs]
        index = VectorStoreIndex.from_documents(docs, embed_model=self.emb_model)
        index.storage_context.persist(persist_dir=self.index_dir)

    def load_index(self):
        storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
        index = load_index_from_storage(storage_context, embed_model=self.emb_model)
        self.index = index
    
    def retrieve(self, quries):
        vector_retriever = self.index.as_retriever(similarity_top_k=self.res_num_per_query)

        documents = []
        seen_titles = set()

        for qid,query in enumerate(quries):
            docs = vector_retriever.retrieve(query)
            if self.debugging:
                for node in docs:
                    print(f"Query: {query} - Score: {node.score:.2f} - {node.text}")

            uniq_docs = []
            for doc in docs:
                if doc.text in seen_titles:
                    continue
                seen_titles.add(doc.text)
                uniq_docs.append(doc)

            for rank, doc in enumerate(uniq_docs):
                if doc.text in self.full_document_maps:
                    ingredients, steps = self.full_document_maps[doc.text]
                    new_doc = RecipeInfo(
                        title=doc.text,
                        ingredients=ingredients,
                        steps=steps,
                        is_original_query=(qid==0),
                        position = rank + 1,
                        score=doc.score
                    )
                    documents.append(new_doc)
                else:
                    print (f'recipe={doc.text} not found')
                    
        if self.has_reranking:
            return self.reranking(quries[0], documents)
        else:
            documents = sorted(documents, key=lambda doc: doc.score, reverse=True)[:self.final_res_num]
        
        return documents
    
    def reranking(self, query, res_list):
        pairs = [[query, doc.rerank_text] for doc in res_list]


        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=128)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            scores = self.reranking_model(**inputs, return_dict=True).logits.view(-1, ).float()
            
        for doc, score in zip(res_list, scores):
            doc.rerank_score = score

        if self.reranking_type == "relevance":
            reranked_results = sorted(res_list, key=lambda doc: doc.rerank_score, reverse=True)[:self.final_res_num]
            for rank, doc in enumerate(reranked_results):
                doc.rerank_position = rank + 1

        if self.reranking_type == "MMR":
            selected_vectors = []
            lambda_param = self.reranking_alpha
            
            remaining_indices = set(range(len(res_list)))

            embeddings = []
            docs = [doc.title for doc in res_list]
            for doc in docs:
                embeddings.append(self.emb_model.get_text_embedding(doc))
            embeddings = np.array(embeddings)

            for num in range(self.final_res_num):
                mmr_scores = []

                for i in remaining_indices:
                    relevance = norm(res_list[i].rerank_score)
                    if selected_vectors:
                        sim_to_selected = cosine_similarity(
                            embeddings[i].reshape(1, -1),
                            np.stack(selected_vectors)
                        ).max()
                    else:
                        sim_to_selected = 0
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * sim_to_selected
                    mmr_scores.append((mmr_score, i))
                max_mmr_scores, selected_idx = max(mmr_scores, key=lambda x: x[0])
                

                res_list[selected_idx].diversity_rerank_nth =  num + 1
                selected_vectors.append(embeddings[selected_idx])
                remaining_indices.remove(selected_idx)

            reranked_results = sorted(res_list, key=lambda doc: doc.diversity_rerank_nth)[:self.final_res_num]
            for rank, doc in enumerate(reranked_results):
                doc.rerank_position = rank + 1

        if  self.reranking_type == "vendi":
            selected_vectors = []
            lambda_param = self.reranking_alpha
            embeddings = []
            docs = [doc.title for doc in res_list]
            for doc in docs:
                embeddings.append(self.emb_model.get_text_embedding(doc))
            embeddings = np.array(embeddings)
            remaining_indices = set(range(len(res_list)))

            for num in range(self.final_res_num):
                vendi_scores = []

                for i in remaining_indices:
                    relevance = norm(res_list[i].rerank_score)
                    if len(selected_vectors) >= 1:
                        temp_vector = np.vstack([np.array(selected_vectors), embeddings[i]])
                        vendi_score = vendi.score_X(temp_vector)
                    else:
                        vendi_score = 0
                    
                    score = lambda_param * relevance + (1 - lambda_param) * vendi_score / (num + 1)
                    vendi_scores.append((score, i))
                max_vendi_scores, selected_idx = max(vendi_scores, key=lambda x: x[0])
                

                res_list[selected_idx].diversity_rerank_nth =  num + 1
                selected_vectors.append(embeddings[selected_idx])
                remaining_indices.remove(selected_idx)

            reranked_results = sorted(res_list, key=lambda doc: doc.diversity_rerank_nth)[:self.final_res_num]
            for rank, doc in enumerate(reranked_results):
                doc.rerank_position = rank + 1

        if self.debugging:
            print("--Reranking--")
            for doc in reranked_results:
                print (doc)

        return reranked_results
        