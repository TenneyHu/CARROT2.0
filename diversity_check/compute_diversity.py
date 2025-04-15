#%%
import warnings
import pandas as pd
import json
import ast
from metrics import clean_ingredients
import numpy as np
from diversity_metrics import compute_global_diversity_from_column, lexical_diversity, msttr, syntactic_diversity, compute_local_diversity_from_column
from lexical_diversity import lex_div as ld
from vendi_score import text_utils
#%%

# if __name__ == "__main__":
    
# configurations 
param_configs = [
    # deterministic sampling
    {"temperature": 0.0000000000001, "top_k": 0, "top_p": 1.0},   # casi greedy
    
    # controlled sampling
    {"temperature": 0.3, "top_k": 20, "top_p": 0.9},    

    # a bit more sampling
    {"temperature": 0.6, "top_k": 40, "top_p": 0.95},

    # more sampling, and add limit top_p
    {"temperature": 0.9, "top_k": 50, "top_p": 0.9},

    # Top-p only (no top_k)
    {"temperature": 0.9, "top_k": 0, "top_p": 0.92},

    # no restrictions in the sampling (big exploration)
    {"temperature": 1.2, "top_k": 0, "top_p": 1.0},

    # common configuration in benchmarks
    {"temperature": 1.0, "top_k": 40, "top_p": 0.85}
]

#%%
config_results = []
lexical_diversity_computation = False
compute_advanced_lexdiv = True
vendi_score_computation = False
compute_global_ingredient_diversity = False 

filename = f'res/adaptation/output_t1e-13_k0_p1.0.json'
with open(filename, 'r') as file:
    data = json.load(file)
df = pd.DataFrame(data)
df['original_text'] = df['src'].apply(lambda x: x['src'])
df_ini = clean_ingredients(df.to_dict('records'), ai_generated=True, list_mode=False, name_column='original_text')
df_ini = pd.DataFrame(df_ini)
print(df_ini.head())
print("Computed clean ingredients for the original recipes.")
print(df_ini['Ingredientes_pr'].to_list()[0])

for param_config in param_configs:
    print(f"***Current configuration: {param_config}")

    temp = param_config['temperature']
    top_k = param_config['top_k']
    top_p = param_config['top_p']
    filename = f'res/adaptation/output_t{temp}_k{top_k}_p{top_p}.json'

    with open(filename, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    # === Preprocessing ===
    df['mod_list'] = df['mod_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['original_text'] = df['src'].apply(lambda x: x['src'])
    df['adapted_texts'] = df['mod_list']
    df['n_adaptations'] = df['adapted_texts'].apply(len)
    df['original_ingredients'] = df_ini['Ingredientes_pr'].to_list()

    # add computation of the original_ingredients
    df = clean_ingredients(df.to_dict('records'), ai_generated=True, list_mode=True, name_column='original_text')
    df = pd.DataFrame(df)
    print(df.head())
    print("Extracted ingredients from adapted texts...")

    print(df.head())
    # Common values (always defined if needed later)
    all_adaptations = [mod for mods in df['adapted_texts'] for mod in mods]
    first_adaptations = [mods[0] for mods in df['adapted_texts'] if mods]
    original_texts = df['original_text'].tolist()

    result_dict = {
        'filename': filename,
        'temperature': temp,
        'top_k': top_k,
        'top_p': top_p,
        'n_samples': len(df),
        'n_adaptations_per_input': df['n_adaptations'].max(),
    }

    if lexical_diversity_computation:
        print("Computing lexical diversity...")

        # Per-input diversity
        df['adapted_lex_diversities'] = df['adapted_texts'].apply(
            lambda mods: [lexical_diversity([mod]) for mod in mods]
        )
        df['per_input_lexical_diversity'] = df['adapted_lex_diversities'].apply(
            lambda scores: np.mean(scores) if scores else 0
        )
        df['per_input_lexical_diversity_std'] = df['adapted_lex_diversities'].apply(
            lambda scores: np.std(scores) if scores else 0
        )

        result_dict.update({
            'original_lexical_diversity': lexical_diversity(original_texts),
            'mean_per_input_lexical_diversity': df['per_input_lexical_diversity'].mean(),
            'mean_per_input_lexical_diversity_std': df['per_input_lexical_diversity_std'].mean(),
            'across_input_lexical_diversity_all': lexical_diversity(all_adaptations),
            'across_input_lexical_diversity_first': lexical_diversity(first_adaptations),
        })

    if compute_advanced_lexdiv:
        print("Computing MSTTR...")
        result_dict.update({
            'original_msttr': msttr(original_texts),
            'across_input_msttr_all': msttr(all_adaptations),
            'across_input_msttr_first': msttr(first_adaptations),
        })

        print("Computing advanced lexical diversity (TTR, MTLD, HDD)...")

        def compute_ld_metrics(text_list):
            tokens = " ".join(text_list).split()
            return {
                'ttr': ld.ttr(tokens),
                'mtld': ld.mtld(tokens),
                'hdd': ld.hdd(tokens),
            }

        original_ld = compute_ld_metrics(original_texts)
        all_adapted_ld = compute_ld_metrics(all_adaptations)
        first_adapted_ld = compute_ld_metrics(first_adaptations)

        result_dict.update({
            'original_hdd': original_ld['hdd'],
            'across_input_hdd_all': all_adapted_ld['hdd'],
            'across_input_hdd_first': first_adapted_ld['hdd'],

            'original_ttr': original_ld['ttr'],
            'across_input_ttr_all': all_adapted_ld['ttr'],
            'across_input_ttr_first': first_adapted_ld['ttr'],

            'original_mtld': original_ld['mtld'],
            'across_input_mtld_all': all_adapted_ld['mtld'],
            'across_input_mtld_first': first_adapted_ld['mtld'],
        })

        # === PER INPUT ===
        print("Computing per-input lexical diversity (TTR, MTLD, HDD)...")

        df['per_input_ld'] = df['adapted_texts'].apply(lambda texts: compute_ld_metrics(texts))

        # Separar las métricas por tipo
        df['per_input_ttr'] = df['per_input_ld'].apply(lambda d: d['ttr'])
        df['per_input_mtld'] = df['per_input_ld'].apply(lambda d: d['mtld'])
        df['per_input_hdd'] = df['per_input_ld'].apply(lambda d: d['hdd'])
        result_dict.update({
            'mean_per_input_ttr': df['per_input_ttr'].mean(),
            'std_per_input_ttr': df['per_input_ttr'].std(),

            'mean_per_input_mtld': df['per_input_mtld'].mean(),
            'std_per_input_mtld': df['per_input_mtld'].std(),

            'mean_per_input_hdd': df['per_input_hdd'].mean(),
            'std_per_input_hdd': df['per_input_hdd'].std(),
        })

        
    if vendi_score_computation:
        print("Computing Vendi Score...")

        # Compute the Vendi Score for all adaptations
        all_adaptations_joined = " ".join(all_adaptations)
        tokens = all_adaptations_joined.split()
        ngram_vs = text_utils.ngram_vendi_score(tokens, ns=[1, 2])
        result_dict.update({
            'vendi_score_ngram_1_2': ngram_vs,
        })

    if compute_global_ingredient_diversity:
        print("Computing global and local ingredient diversity...")


        df['country'] = 'Spain'  # si no tienes una columna de país real

        # === GLOBAL ORIGINAL ===
        global_diversity_original = compute_global_diversity_from_column(df, column='original_ingredients', mode="original")
        mean_global_div_original = np.mean([r['global'] for r in global_diversity_original])
        result_dict.update({
            'original_global_diversity': mean_global_div_original,
        })

        # === GLOBAL ADAPTED ===
        global_diversity_stats = compute_global_diversity_from_column(df, column='Ingredientes_pr', mode="adapted")
        mean_global_div = np.mean([r['global'] for r in global_diversity_stats])
        result_dict.update({
            'adapted_global_diversity': mean_global_div,
        })

        # === GLOBAL FIRST ADAPTATION ===
        df['Ingredientes_pr_first'] = df['Ingredientes_pr'].apply(
            lambda lista: lista[0] if isinstance(lista, list) and len(lista) > 0 else []
        )
        global_diversity_first = compute_global_diversity_from_column(df, column='Ingredientes_pr_first', mode="adapted_first")
        mean_global_div_first = np.mean([r['global'] for r in global_diversity_first])
        result_dict.update({
            'adapted_global_diversity_first': mean_global_div_first,
        })

        df['mode'] = 'original'
        # === LOCAL ORIGINAL ===
        local_diversity_original = compute_local_diversity_from_column(df, column='original_ingredients', mode_label="original")
        mean_local_div_original = np.mean([r['entropy'] for r in local_diversity_original])
        result_dict.update({
            'original_local_diversity': mean_local_div_original,
        })

        # === LOCAL ADAPTED ===
        local_diversity_stats = compute_local_diversity_from_column(df, column='Ingredientes_pr', mode_label="adapted")
        mean_local_div = np.mean([r['entropy'] for r in local_diversity_stats])
        result_dict.update({
            'adapted_local_diversity': mean_local_div,
        })

        # === LOCAL FIRST ADAPTATION ===
        local_diversity_first = compute_local_diversity_from_column(df, column='Ingredientes_pr_first', mode_label="adapted_first")
        mean_local_div_first = np.mean([r['entropy'] for r in local_diversity_first])
        result_dict.update({
            'adapted_local_diversity_first': mean_local_div_first,
        })



    # if compute_global_ingredient_diversity:
    #     print("Computing global ingredient diversity...")

    #     try:
    #         df['country'] = 'Spain'  # si no tienes country real

    #         global_diversity_original = compute_global_diversity_from_column(df, column='original_ingredients', mode="adapted")
    #         diversity_counts = [r['global'] for r in global_diversity_original]
    #         mean_global_div = np.mean(diversity_counts)
    #         result_dict.update({
    #             'original_global_diversity': mean_global_div,
    #         })

    #         global_diversity_stats = compute_global_diversity_from_column(df, column='Ingredientes_pr', mode="adapted")
    #         diversity_counts = [r['global'] for r in global_diversity_stats]
    #         mean_global_div = np.mean(diversity_counts)

    #         result_dict.update({
    #             'adapted_global_diversity': mean_global_div,
    #         })

    #         # only first generation of each input
    #         # Extraer solo los ingredientes de la primera adaptación por receta
    #         df['Ingredientes_pr_first'] = df['Ingredientes_pr'].apply(
    #             lambda lista_de_listas: lista_de_listas[0] if isinstance(lista_de_listas, list) and len(lista_de_listas) > 0 else []
    #         )
    #         global_diversity_first = compute_global_diversity_from_column(df, column='Ingredientes_pr_first', mode="adapted_first")
    #         diversity_first_counts = [r['global'] for r in global_diversity_first]
    #         mean_global_div_first = np.mean(diversity_first_counts)
    #         result_dict.update({
    #             'adapted_global_diversity_first': mean_global_div_first,
    #         })
            
    #     except Exception as e:
    #         print(f"Could not compute global ingredient diversity: {e}")

    


    config_results.append(result_dict)

# === Save results ===
df_summary = pd.DataFrame(config_results)
df_summary.to_csv('res/adaptation/diversity_results.csv', index=False)


# %%
