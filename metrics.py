import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def load_evaluate_res(filepath, cutoff):
    recipes = []
    scores = []
    with open(filepath, "r") as file:
        for line in file:
            pos = int(line.strip().split("\t")[2])
            score =float(line.strip().split("\t")[3])
            recipe = line.strip().split("\t")[4]
            if pos <= cutoff:
                scores.append(score)
                recipes.append(recipe)
    return scores,recipes

def calc_semantic_diversity(model, texts):
    embeddings = model.encode(texts)
    similarity_matrix = cosine_similarity(embeddings)
    upper_triangular_values = similarity_matrix[np.triu_indices(len(texts), k=1)]
    mean_similarity = np.mean(upper_triangular_values)
    
    # Diversity is 1 - similarity (higher means more diverse)
    return 1 - mean_similarity

cos_score_source = []
semantic_diversity = []
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-es', trust_remote_code=True).to('cuda')

for k in range(1,21):
    scores, recipes = load_evaluate_res("./res/retrieval_res", k)
    #avg cos score
    avg_score = sum(scores) / len(scores) 
    cos_score_source.append(avg_score)
   
    semantic_score = calc_semantic_diversity(model, recipes)
    semantic_diversity.append(semantic_score)
    print (k, semantic_score)


plt.plot(range(1, 21), cos_score_source, marker='o')
plt.title('Average Relevance Score vs. #Retrieval Results')
plt.xlabel('#Retrieval Results')
plt.ylabel('Average Relevance Score')
#plt.ylim(0, 1)

plt.grid(True)
plt.savefig('./figures/average_score_vs_cutoff.png')

