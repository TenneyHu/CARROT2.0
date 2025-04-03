from transformers import AutoModel
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from metrics_utils import calc_global_diversity, calc_avg_semantic_diversity
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='jinaai/jina-embeddings-v2-base-es',
                        help='Name of the model to load')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to load the model on')
    parser.add_argument('--results_file', type=str, default='./res/res1',
                        help='File path for retrieval results')
    parser.add_argument('--results_per_query', type=int, default=21,
        help='return results per query')
    parser.add_argument('--nums_query_rewriting', type=int, default=3,
        help='nums of query rewriting')
    return parser.parse_args()

def load_model(model_name, device):
    return AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

def parse_results(dir,ingredients_pos=7):

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
            ingredients = ",".join(parts[ingredients_pos:]).strip().split("=")[1].strip('"')

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

def plot_metrics_multi(groups, filename):
    """
    groups: List of dicts with keys:
        - 'label': name of the group
        - 'source_relevance_score': list of floats
        - 'semantic_diversity': list of floats
        - 'global_diversity': list of floats
        - 'ks': List of integers (x-axis)
    filename: Output image file
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))  # 3 subplots, vertical

    colors = cm.get_cmap('tab10', len(groups))

    for idx, group in enumerate(groups):
        label = group['label']
        relevance = group['source_relevance_score']
        sem_div = group['semantic_diversity']
        glob_div = group['global_diversity']
        ks = group['ks']
        color = colors(idx)

        # Relevance Plot
        ax1.plot(ks, relevance, marker='o', linestyle='-', color=color, label=label)

        # Semantic Diversity Plot
        ax2.plot(ks, sem_div, marker='s', linestyle='--', color=color, label=label)

        # Global Diversity Plot
        ax3.plot(ks, glob_div, marker='^', linestyle='--', color=color, label=label)

    # Setup for Relevance
    ax1.set_title('Average Relevance Score vs. Top-k')
    ax1.set_xlabel('#Retrieval Results (Top-k)')
    ax1.set_ylabel('Average Relevance Score')
    ax1.set_xticks(ks)
    ax1.grid(True)
    ax1.legend()

    # Setup for Semantic Diversity
    ax2.set_title('Semantic Diversity vs. Top-k')
    ax2.set_xlabel('#Retrieval Results (Top-k)')
    ax2.set_ylabel('Semantic Diversity')
    ax2.set_xticks(ks)
    ax2.grid(True)
    ax2.legend()

    # Setup for Global Diversity
    ax3.set_title('Global Diversity vs. Top-k')
    ax3.set_xlabel('#Retrieval Results (Top-k)')
    ax3.set_ylabel('Global Diversity')
    ax3.set_xticks(ks)
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def calc_metrics(model, filtered_result, k_range, relevance_score= "reranking", filter="retrieval"):
    source_relevance_score = []
    semantic_diversity = []
    global_diversity = []

    for k in tqdm(range(1, k_range+1)):
        if filter == 'retrieval':
            results = filtered_result[filtered_result["retrieve_position"] <= k]
        else:
            results = filtered_result[filtered_result["rerank_position"] <= k]
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

def calc_original_query(args, model, results):
    filtered_result = results[results["from_original_query"] == True]
    k_range = args.results_per_query
    source_relevance_score, semantic_diversity, global_diversity = calc_metrics(model, filtered_result, k_range)
    ks = list(range(1, k_range + 1))
    res = {
        'label': 'Base Retrieval',
        'source_relevance_score': source_relevance_score,
        'semantic_diversity': semantic_diversity,
        'global_diversity': global_diversity,
        'ks': ks
    }
    return res

def calc_reranking(args, model, results, label):
    k_range = args.results_per_query
    source_relevance_score, semantic_diversity, global_diversity = calc_metrics(model, results, k_range, relevance_score= "reranking", filter="reranking")
    ks = list(range(1, k_range + 1))
    res = {
        'label': label,
        'source_relevance_score': source_relevance_score,
        'semantic_diversity': semantic_diversity,
        'global_diversity': global_diversity,
        'ks': ks
    }
    return res

def calc_rewriting(args, model, results):
    k_range = 7
    source_relevance_score, semantic_diversity, global_diversity = calc_metrics(model, results, k_range)
    ks = list(range(3,22,3))
    res = {
        'label': 'Rewriting',
        'source_relevance_score': source_relevance_score,
        'semantic_diversity': semantic_diversity,
        'global_diversity': global_diversity,
        'ks': ks
    }
    return res

def calc_singlek_metric(model, filtered_result, k):
    results = filtered_result[filtered_result["rerank_position"] <= k]
    qid_to_titles = results.groupby("qid")["title"].apply(list).to_dict()
    ingredients = results["ingredients"].tolist()
    scores = results["rerank_score"].tolist()
    source_relevance_score = sum(scores) / len(scores)
    semantic_diversity = calc_avg_semantic_diversity(model, qid_to_titles)
    global_diversity = calc_global_diversity(ingredients)
    return source_relevance_score, semantic_diversity, global_diversity

def calc_rewriting(model, label, results, k):
    k_range = 7
    source_relevance_score, semantic_diversity, global_diversity = calc_singlek_metric(model, results, k)
    ks = list(range(3,22,3))
    res = {
        'label': label,
        'source_relevance_score': source_relevance_score,
        'semantic_diversity': semantic_diversity,
        'global_diversity': global_diversity,
        'ks': ks
    }
    return res


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model_name, args.device)
    

    # original query
    '''
    res = []
    results = parse_results(args.results_file)
    #res.append(calc_original_query(args, model, results))
    #res.append(calc_rewriting(args, model, results))
    res.append(calc_reranking(args, model, results, "relevance reranking"))

    results = parse_results("./res/res",8)
    res.append(calc_reranking(args, model, results, "mmr diversity reranking"))


    results = parse_results("./res/vendi",8)
    res.append(calc_reranking(args, model, results, "vendi diversity reranking"))
    plot_metrics_multi(res, './figures/reranking_tradeoff.png')
    '''

    #parse with different lambda
 
for k in [3, 5, 7, 10]:
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    metrics = {
        "mmr": {"source_relevance_score": [], "semantic_diversity": [], "global_diversity": []},
        "vendi": {"source_relevance_score": [], "semantic_diversity": [], "global_diversity": []}
    }

    colors = {
        "mmr": "blue",
        "vendi": "orange"
    }

    for algorithm in ["mmr", "vendi"]:
        for alpha in alphas:
            filename = f"./res/{algorithm}_alpha{alpha}"
            results = parse_results(filename, 8)
            res1, res2, res3 = calc_singlek_metric(model, results, k)
            metrics[algorithm]["source_relevance_score"].append(res1)
            metrics[algorithm]["semantic_diversity"].append(res2)
            metrics[algorithm]["global_diversity"].append(res3)

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Source Relevance Score
    axs[0].plot(alphas, metrics["mmr"]["source_relevance_score"], marker='o', linestyle='-', label='MMR', color=colors["mmr"])
    axs[0].plot(alphas, metrics["vendi"]["source_relevance_score"], marker='s', linestyle='--', label='Vendi', color=colors["vendi"])
    axs[0].set_xlabel('Alpha')
    axs[0].set_ylabel('Source Relevance Score')
    axs[0].set_title(f'Source Relevance Score for k={k}')
    axs[0].legend()

    # Semantic Diversity
    axs[1].plot(alphas, metrics["mmr"]["semantic_diversity"], marker='o', linestyle='-', label='MMR', color=colors["mmr"])
    axs[1].plot(alphas, metrics["vendi"]["semantic_diversity"], marker='s', linestyle='--', label='Vendi', color=colors["vendi"])
    axs[1].set_xlabel('Alpha')
    axs[1].set_ylabel('Semantic Diversity')
    axs[1].set_title(f'Semantic Diversity for k={k}')
    axs[1].legend()

    # Global Diversity
    axs[2].plot(alphas, metrics["mmr"]["global_diversity"], marker='o', linestyle='-', label='MMR', color=colors["mmr"])
    axs[2].plot(alphas, metrics["vendi"]["global_diversity"], marker='s', linestyle='--', label='Vendi', color=colors["vendi"])
    axs[2].set_xlabel('Alpha')
    axs[2].set_ylabel('Global Diversity')
    axs[2].set_title(f'Global Diversity for k={k}')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f'./figures/compare_metrics_alpha_k{k}.png')
    plt.close()