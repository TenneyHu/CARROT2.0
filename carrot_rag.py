from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from llama_index.llms.ollama import Ollama
from llama_index.readers.faiss import FaissReader
from llama_index.core import SummaryIndex
import faiss
import pickle
import ollama

def recipe_title_generate(recipe):

    response = ollama.chat(model="llama3.1", messages=[
        {"role": "system", "content": "Here is a recipe without title; please create a short Spanish title for the recipe."},
        {"role": "user", "content": f"The recipe is {recipe}"},
        {"role": "user", "content": f" Please only output the recipe title, do not output any other content."}
    ])

    
    title = response["message"]["content"]
    return title 

def recipe_title_rewrite(ori_title):
    response = ollama.chat(model="llama3.1", messages=[
        {"role": "system", "content": "Please rewrite the title of this recipe to align with Spanish recipe naming conventions and dietary habits."},
        {"role": "user", "content": f"The recipe is {ori_title}"},
        {"role": "user", "content": f" Please only output the recipe title, do not output any other content."}
    ])
    title = response["message"]["content"]
    return title

def save_index(docs, embedding_model, dir, device="cuda:0"):

    print ("--indexing start--")
    batch_size = 2048
    model = SentenceTransformer(embedding_model, device=device)
    embeddings = model.encode(docs, batch_size = batch_size)  
    embeddings = np.array(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dimension) 

    for i in tqdm(range(0, len(embeddings), batch_size), desc="FAISS Indexing", unit="batch"):
        index.add(embeddings[i:i+batch_size])

    id_to_text_map = {i: text for i, text in enumerate(docs)}
    faiss.write_index(index, dir+ "faiss_index.bin")
    with open(dir + "id_to_text_map.pkl", "wb") as f:
        pickle.dump(id_to_text_map, f)

def load_index(dir):
    index = faiss.read_index(dir + "faiss_index.bin") 
    with open(dir + "id_to_text_map.pkl", "rb") as f:
        id_to_text_map = pickle.load(f)     
    return index, id_to_text_map



class CarrotQueryEngine:
    def __init__(self, emb_model, index_dir, full_document_maps, llm, switch_query_rewrite = 0, debugging = 0, k = 10):
        self.emb_model = SentenceTransformer(emb_model)
        self.full_document_maps = full_document_maps
        index, self.id_to_text_map = load_index(index_dir)
        self.reader = FaissReader(index)
        self.llm = Ollama(model=llm, request_timeout=300.0)
        self.query_rewrite = switch_query_rewrite
        self.debugging = debugging
        self.k = k

    def query_processing(self, ori_title, ori_content):
        if self.query_rewrite:
            generated_query = recipe_title_generate(ori_content)
            rewrited_query = recipe_title_rewrite(ori_title)
            res = [ori_title, generated_query, rewrited_query]
        else:
            res = [ori_title]
        
        if self.debugging and self.query_rewrite:
            print("Original query: ", ori_title)
            print("Generated query: ", generated_query) 
            print("Rewritten query: ", rewrited_query)

        return res
    
    def retrieve(self, quries):
        for query in quries:
            query_emb = self.emb_model.encode([query])
            docs = self.reader.load_data(query=query_emb, id_to_text_map=self.id_to_text_map, k=self.k)
            if self.debugging:
                for i, doc in enumerate(docs):
                    print(f"Query: {query} Rank {i+1}: {doc.text}")

            if 'documents' not in locals():
                documents = docs
            else:
                documents.extend(docs)

        for document in documents:
            document.text = self.full_document_maps[document.text]
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine(llm = self.llm)

        return query_engine
