
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import ollama
from llama_index.core.schema import NodeWithScore
from llama_index.core.data_structs import Node

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

def carrot_query_processing(ori_title, ori_content, query_rewrite, debugging):
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
    
class CarrotRetriever():
    def __init__(self, emb_model, index_dir, full_document_maps, debugging = 0, 
                    reranking = 0, res_num_per_query = 10, final_res_num = 5):
        self.emb_model = HuggingFaceEmbedding(model_name = emb_model, trust_remote_code = True)
        self.full_document_maps = full_document_maps
        self.index_dir = index_dir
        self.debugging = debugging
        self.index = None
        self.has_reranking = reranking
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
        print ("---Retrieval---")
        vector_retriever = self.index.as_retriever(similarity_top_k=self.res_num_per_query)
        for query in quries:
            docs = vector_retriever.retrieve(query)
            if self.debugging:
                for node in docs:
                    print(f"Query: {query} - Score: {node.score:.2f} - {node.text}")
            if 'documents' not in locals():
                documents = docs
            else:
                documents.extend(docs)

        updated_documents = []
        for doc in documents:
            new_text = self.full_document_maps.get(doc.text, doc.text)
            new_doc = NodeWithScore(node=Node(text=new_text), score=doc.score)
            updated_documents.append(new_doc)

        if self.has_reranking:
            return self.reranking(updated_documents)
        else:
            updated_documents = sorted(updated_documents, key=lambda doc: doc.score, reverse=True)[:self.final_res_num]
        
        return updated_documents
    
    def reranking(self, res_list):
        reranked_results = sorted(res_list, key=lambda doc: doc.score, reverse=True)[:self.final_res_num]
        if self.debugging:
            print("--Reranking--")
            for doc in reranked_results:
                print(f"Score: {doc.score:.2f} - {doc.text}")
        return reranked_results
        