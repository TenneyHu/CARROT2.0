#%%
import warnings
import pandas as pd
import json
import ast
import numpy as np
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

from metrics import clean_ingredients
from diversity_metrics import (
    avg_pairwise_jaccard_diversity,
    calc_avg_semantic_diversity,
    compute_global_diversity_from_column,
    compute_input_diversity,
    compute_self_bleu,
    compute_self_bleu_parallel,
    lexical_diversity,
    msttr,
    syntactic_diversity,
    compute_local_diversity_from_column
)
from lexical_diversity import lex_div as ld
from vendi_score import text_utils
from sentence_transformers import SentenceTransformer

# ========== PARAMETERS ==========
param_configs = [
    {"temperature": 0.0000000000001, "top_k": 0, "top_p": 1.0},   # casi greedy
    {"temperature": 0.0000000000001, "top_k": 50, "top_p": 0.9},   # casi greedy
    {"temperature": 0.3, "top_k": 20, "top_p": 0.9},
    {"temperature": 0.3, "top_k": 50, "top_p": 0.9},       
    {"temperature": 0.6, "top_k": 40, "top_p": 0.95},
    {"temperature": 0.9, "top_k": 50, "top_p": 0.9},
    {"temperature": 0.9, "top_k": 0, "top_p": 0.92},
    {"temperature": 1.0, "top_k": 40, "top_p": 0.85},
    {"temperature": 1.0, "top_k": 0, "top_p": 0.85},
    {"temperature": 1.0, "top_k": 20, "top_p": 0.85},
    {"temperature": 1.2, "top_k": 0, "top_p": 1.0},
    {"temperature": 1.2, "top_k": 20, "top_p": 1.0},
    {"temperature": 1.2, "top_k": 40, "top_p": 1.0},
    {"temperature": 1.2, "top_k": 40, "top_p": 0.85},
    {"temperature": 1.2, "top_k": 50, "top_p": 1.0},
]

# ========== UTILS ==========
def load_json_to_df(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['original_text'] = df['src'].apply(lambda x: x['src'])
    df['mod_list'] = df['mod_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['adapted_texts'] = df['mod_list']
    df['n_adaptations'] = df['adapted_texts'].apply(len)
    return df

def compute_lexical_diversity_metrics(df):
    first_adaptations = [mods[0] for mods in df['adapted_texts'] if mods]
    df['adapted_lex_diversities'] = df['adapted_texts'].apply(
        lambda mods: [lexical_diversity([mod]) for mod in mods]
    )
    df['per_input_lexical_diversity'] = df['adapted_lex_diversities'].apply(
        lambda scores: np.mean(scores) if scores else 0
    )
    df['per_input_lexical_diversity_std'] = df['adapted_lex_diversities'].apply(
        lambda scores: np.std(scores) if scores else 0
    )
    metrics = {
        'across_input_adapted_ttr': lexical_diversity(first_adaptations),
        'across_input_adapted_msttr': msttr(first_adaptations),
        'mean_per_input_lexical_diversity': df['per_input_lexical_diversity'].mean(),
        'mean_per_input_lexical_diversity_std': df['per_input_lexical_diversity_std'].mean(),
    }
    return metrics

import numpy as np
from scipy.spatial.distance import pdist
def compute_semantic_diversity_metrics(df, original_recipes, model):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    
    all_texts = set()
    for mods in df['adapted_texts']:
        all_texts.update(mods)
    all_texts.update(original_recipes)
    all_texts = list(all_texts)
    text2embedding = {}
    batch_size = 4
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, device=device)
        for t, emb in zip(batch, embeddings):
            text2embedding[t] = emb
    print("Text embeddings computed!")

    def avg_cosine_distance(texts):
        if len(texts) < 2:
            return 0.0
        vecs = np.stack([text2embedding[t] for t in texts])
        dists = pdist(vecs, metric="cosine")
        return float(np.mean(dists))
    
    # Per-input (average over inputs, each is K generations)
    df['per_input_semantic_diversity'] = df['adapted_texts'].apply(avg_cosine_distance)
    per_input_semdiv_mean = df['per_input_semantic_diversity'].mean()
    per_input_semdiv_std = df['per_input_semantic_diversity'].std()
    print("Cosine distances computed!")


    first_adaptations = [mods[0] for mods in df['adapted_texts'] if mods]
    across_input_adapted_semdiv = avg_cosine_distance(first_adaptations)  # semantic
    print("Across input semantic diversity computed!")


    across_input_original_semdiv = avg_cosine_distance(original_recipes)
    print("Across input original semantic diversity computed!")

    df['per_input_self_bleu'] = df['adapted_texts'].apply(
        lambda ts: compute_self_bleu(ts) if len(ts) > 1 else None)
    per_input_self_bleu_mean = df['per_input_self_bleu'].mean()
    per_input_self_bleu_std = df['per_input_self_bleu'].std()
    print("Per input Self-bleu values computed!")

    across_input_original_self_bleu = compute_self_bleu_parallel(original_recipes) if len(original_recipes) > 1 else None
    print("Across input original Self-bleu values computed!")
    # Don't include "all_adapted_texts" for across-input: it's not in the paper's formulation

    across_input_adapted_self_bleu = compute_self_bleu_parallel(first_adaptations) if len(first_adaptations) > 1 else None
    print("Across input Self-bleu values computed!")
    return {
        'per_input_adapted_semdiv_mean': per_input_semdiv_mean,
        'per_input_adapted_semdiv_std': per_input_semdiv_std,
        'across_input_adapted_semdiv': across_input_adapted_semdiv,
        'across_input_original_semdiv': across_input_original_semdiv,
        'per_input_self_bleu_mean': per_input_self_bleu_mean,
        'per_input_self_bleu_std': per_input_self_bleu_std,
        'across_input_adapted_self_bleu': across_input_adapted_self_bleu,
        'across_input_original_self_bleu': across_input_original_self_bleu,
    }



def compute_ingredient_diversity_metrics(df):
    metrics = {}

    # GLOBAL DIVERSITY
    # ---> Across input global diversity (original data)
    global_diversity_original = compute_global_diversity_from_column(df, column='original_ingredients', mode="original")
    metrics['across_input_global_diversity_original'] = np.mean([r['global'] for r in global_diversity_original])

    # ---> Across input global diversity (adapted data) here 
    # 1. We take the first adaptation only. 
    df['Ingredientes_pr_first'] = df['Ingredientes_pr'].apply(
        lambda lista: lista[0] if isinstance(lista, list) and len(lista) > 0 else []
    )
    global_diversity_first = compute_global_diversity_from_column(df, column='Ingredientes_pr_first', mode="adapted")
    metrics['across_input_global_diversity_adapted'] =  np.mean([r['global'] for r in global_diversity_first])

    # ---> Per input global diversity (adapted data, because in original data we have only one recipe so no diversity)
    # I think it doesnt make sense to compute the global diversity for inside one input recipe, because of the definition of global diversity.
    # i think it is better to study the diversity here as ingredient overlap between the different adaptations of the same recipe. 
    df['pairwise_ingredient_diversity'] = df['mod_list'].apply(
        lambda lsts: avg_pairwise_jaccard_diversity(lsts) if len(lsts) > 1 else 1.0
    )
    df['pairwise_ingredient_diversity'] = df['pairwise_ingredient_diversity'].replace([np.inf, -np.inf], 0)
    metrics['per_input_pairwise_ingredient_diversity'] = df['pairwise_ingredient_diversity'].mean()
    # Additionally, we ccomplete the 
    df['avg_ingredient_length'] = df['Ingredientes_pr'].apply(lambda ings: np.mean([len(lst) for lst in ings]))
    df['std_ingredient_length'] = df['Ingredientes_pr'].apply(lambda ings: np.std([len(lst) for lst in ings]))

    # Overall (across all adaptations in the dataset)
    all_lengths = [len(lst) for recipe in df['Ingredientes_pr'] for lst in recipe]
    metrics['overall_avg_length'] = np.mean(all_lengths)
    metrics['overall_std_length'] = np.std(all_lengths)

    # LOCAL DIVERSITY
    local_diversity_original = compute_local_diversity_from_column(df, column='original_ingredients', mode_label="original")
    metrics['across_input_local_diversity_original'] = np.mean([r['entropy'] for r in local_diversity_original])

    local_diversity_first = compute_local_diversity_from_column(df, column='Ingredientes_pr_first', mode_label="adapted")
    metrics['across_input_global_diversity_adapted'] =  np.mean([r['entropy'] for r in local_diversity_first])
    
    input_diversities = [
        compute_input_diversity(row['Ingredientes_pr'])
        if isinstance(row['Ingredientes_pr'], list) and all(isinstance(opt, list) for opt in row['Ingredientes_pr']) and len(row['Ingredientes_pr']) > 1
        else np.nan
        for _, row in df.iterrows()
    ]
    mean_per_input_diversity = np.nanmean(input_diversities)
    metrics['mean_per_input_local_diversity'] = mean_per_input_diversity


    return metrics

# ------------------------------------------------------------------------------------
#%%
# ========== MAIN SCRIPT ==========
diversity_computation = {'LEXICAL': False, 'SEMANTIC': True, 'SYNTACTIC': False, 'INGREDIENT': False}
for diversity_type in diversity_computation:
    print(f"Computing {diversity_type} diversity...")
    filename_results = f'res/adaptation/diversity_results_{diversity_type.lower()}.csv'
    filename_partial_results = f'res/adaptation/diversity_partial_results_{diversity_type.lower()}.csv'

    if not diversity_computation[diversity_type]:
        print(f"[Skipping {diversity_type} diversity computation]")
        continue

    # --- Carga y extracción de ingredientes originales SOLO UNA VEZ ---
    first_config = param_configs[0]
    first_filename = f'res/adaptation/output_t{first_config["temperature"]}_k{first_config["top_k"]}_p{first_config["top_p"]}.json'
    df0 = load_json_to_df(first_filename)
    df0['country'] = 'Spain'
    df0_cleaned = clean_ingredients(df0.to_dict('records'), ai_generated=True, list_mode=False, name_column='original_text')
    df0_cleaned = pd.DataFrame(df0_cleaned)
    original_ingredients_list = df0_cleaned['Ingredientes_pr'].tolist()
    original_recipes = df0['original_text'].tolist()
    original_ttr = lexical_diversity(original_recipes)
    original_msttr = msttr(original_recipes)
    # --- FIN Carga ingredientes originales ---

    config_results = []

    for iteration, param_config in enumerate(param_configs):
        print(f"***Current configuration: {param_config}")

        # we need to save the config for posterior evaluation analysis
        temp = param_config['temperature']
        top_k = param_config['top_k']
        top_p = param_config['top_p']

        filename = f'res/adaptation/output_t{temp}_k{top_k}_p{top_p}.json'
        df = load_json_to_df(filename)
        df['country'] = 'Spain'
        df_cleaned = clean_ingredients(df.to_dict('records'), ai_generated=True, list_mode=True, name_column='original_text')
        df_cleaned = pd.DataFrame(df_cleaned)

        # --- ASIGNA ingredientes originales alineados ---
        df_cleaned['original_ingredients'] = original_ingredients_list
        # ------------------------------------------------

        # Para debug visual
        # print(df_cleaned[['Ingredientes_pr', 'original_ingredients']].head())

        if diversity_type == 'LEXICAL':
            lex_metrics = compute_lexical_diversity_metrics(df)
            result_dict = {
                'filename': filename,
                'temperature': temp,
                'top_k': top_k,
                'top_p': top_p,
                'n_samples': len(df),
                'n_adaptations_per_input': df['n_adaptations'].max(),
                'across_input_original_ttr': original_ttr,
                'across_input_original_msttr': original_msttr,
                **lex_metrics
            }

        if diversity_type == 'SEMANTIC':
            model = SentenceTransformer("jinaai/jina-embeddings-v2-base-es", trust_remote_code=True)
            sem_metrics = compute_semantic_diversity_metrics(df, original_recipes, model)
            result_dict = {
                'filename': filename,
                'temperature': temp,
                'top_k': top_k,
                'top_p': top_p,
                'n_samples': len(df),
                'n_adaptations_per_input': df['n_adaptations'].max(),
                **sem_metrics
            }

        if diversity_type == 'SYNTACTIC':
            pass

        if diversity_type == 'INGREDIENT':
            print("\tComputing ingredient diversity metrics...")
            ingredient_metrics = compute_ingredient_diversity_metrics(df_cleaned)
            result_dict = {
                'filename': filename,
                'temperature': temp,
                'top_k': top_k,
                'top_p': top_p,
                'n_samples': len(df_cleaned),
                'n_adaptations_per_input': df['n_adaptations'].max(),
                **ingredient_metrics
            }

        config_results.append(result_dict)

        # save intermediate results
        df_intermediate = pd.DataFrame(config_results)
        df_intermediate.to_csv(''+filename_partial_results, index=False)
        print(f"Results saved to {filename_partial_results}")

    # Save final results
    df_summary = pd.DataFrame(config_results)
    df_summary.to_csv(filename_results, index=False)

# %%
