from transformers import AutoModel
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics_utils import calc_global_diversity, calc_avg_semantic_diversity
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='jinaai/jina-embeddings-v2-base-es',
                        help='Name of the model to load')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to load the model on')
    parser.add_argument('--results_file', type=str, default='./res/retrieval_res',
                        help='File path for retrieval results')
    parser.add_argument('--results_per_query', type=int, default=21,
        help='return results per query')
    parser.add_argument('--nums_query_rewriting', type=int, default=3,
        help='nums of query rewriting')
    return parser.parse_args()

def load_model(model_name, device):
    return AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

def parse_results(dir):

    data = {
        "qid": [],
        "title": [],
        "from_original_query": [],
        "retrieve_position": [],
        "rerank_position": [],
        "score": [],
        "rerank_score": [],
        "ingredients": []
    }

    with open(dir, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split(", ")
            qid = int(parts[0].split("=")[1])
            title = parts[1].split("=")[1].strip("'")
            from_original_query = parts[2].split("=")[1] == 'True'
            retrieve_position = int(parts[3].split("=")[1])
            rerank_position = int(parts[4].split("=")[1])
            score = float(parts[5].split("=")[1])
            rerank_score = float(parts[6].split("=")[1])
            ingredients = ",".join(parts[7:]).strip().split("=")[1].strip('"')

            data["qid"].append(qid)
            data["title"].append(title)
            data["from_original_query"].append(from_original_query)
            data["retrieve_position"].append(retrieve_position)
            data["rerank_position"].append(rerank_position)
            data["score"].append(score)
            data["rerank_score"].append(rerank_score)
            data["ingredients"].append(ingredients)

    df = pd.DataFrame(data)
    # Normalize the rerank_score column to a range of 0 to 1
    min_score = df["rerank_score"].min()
    max_score = df["rerank_score"].max()
    df["rerank_score"] = (df["rerank_score"] - min_score) / (max_score - min_score)
    return df

def calc_metrics(model, filtered_result, k_range, relevance_score= "reranking"):
    source_relevance_score = []
    semantic_diversity = []
    global_diversity = []

    for k in tqdm(range(1, k_range+1)):
        results = filtered_result[filtered_result["retrieve_position"] <= k]
        # Ensure results are unique by title
        #results = results.drop_duplicates(subset="title")

        qid_to_titles = results.groupby("qid")["title"].apply(list).to_dict()

        ingredients = results["ingredients"].tolist()
        
        if relevance_score == "reranking":
            scores = results["rerank_score"].tolist()
        else:
            scores = results["score"].tolist()

        avg_score = sum(scores) / len(scores) 
        source_relevance_score.append(avg_score)

        res = calc_avg_semantic_diversity(model, qid_to_titles)
        semantic_diversity.append(res)

        res = calc_global_diversity(ingredients)
        global_diversity.append(res)

    return source_relevance_score, semantic_diversity, global_diversity

def plot_metrics(source_relevance_score, semantic_diversity, global_diversity, ks, filename):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot semantic diversity vs source relevance score
    ax1.plot(ks, source_relevance_score, marker='o', linestyle='-', color='tab:blue', label='Average Relevance Score')
    ax1.set_xlabel('#Retrieval Results (Top-k)')
    ax1.set_ylabel('Average Relevance Score', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(ks)
    ax1.grid(True)

    ax1_2 = ax1.twinx()
    ax1_2.plot(ks, semantic_diversity, marker='s', linestyle='--', color='tab:orange', label='Semantic Diversity')
    ax1_2.set_ylabel('Semantic Diversity', color='tab:orange')
    ax1_2.tick_params(axis='y', labelcolor='tab:orange')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_1_2, labels_1_2 = ax1_2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_1_2, labels_1 + labels_1_2, loc='upper right')
    ax1.set_title('Semantic Diversity vs. Average Relevance Score')

    # Plot global diversity vs source relevance score
    ax2.plot(ks, source_relevance_score, marker='o', linestyle='-', color='tab:blue', label='Average Relevance Score')
    ax2.set_xlabel('#Retrieval Results (Top-k)')
    ax2.set_ylabel('Average Relevance Score', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_xticks(ks)
    ax2.grid(True)

    ax2_2 = ax2.twinx()
    ax2_2.plot(ks, global_diversity, marker='s', linestyle='--', color='tab:green', label='Global Diversity')
    ax2_2.set_ylabel('Global Diversity', color='tab:green')
    ax2_2.tick_params(axis='y', labelcolor='tab:green')

    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2_2, labels_2_2 = ax2_2.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_2_2, labels_2 + labels_2_2, loc='upper right')
    ax2.set_title('Global Diversity vs. Average Relevance Score')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)

def calc_original_query(args, model, results):
    filtered_result = results[results["from_original_query"] == True]
    k_range = args.results_per_query
    source_relevance_score, semantic_diversity, global_diversity = calc_metrics(model, filtered_result, k_range)
    ks = list(range(1, k_range + 1))
    plot_metrics(source_relevance_score, semantic_diversity, global_diversity, ks, "figures/retrieve_only.png")

def calc_rewriting(args, model, results):
    filtered_result = results
    k_range = 7
    source_relevance_score, semantic_diversity, global_diversity = calc_metrics(model, filtered_result, k_range)
    ks = list(range(3,22,3))
    print (ks)
    plot_metrics(source_relevance_score, semantic_diversity, global_diversity, ks, "figures/retrieve_only.png")

if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model_name, args.device)
    results = parse_results(args.results_file)

    # original query
    #calc_original_query(args, model, results)

    #rewriting
    calc_rewriting(args, model, results)