#%%
import warnings
import pandas as pd
import json
import ast
import numpy as np
from diversity_metrics import lexical_diversity, msttr, syntactic_diversity
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
compute_advanced_lexdiv = False
vendi_score_computation = True

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
        
        # Concatenate all original texts into a single string for analysis
        all_text_joined = " ".join(original_texts)
        tokens = all_text_joined.split()  # or use nltk.word_tokenize if needed
        
        ttr_value = ld.ttr(tokens)
        mtld_value = ld.mtld(tokens)
        hdd_value = ld.hdd(tokens)


        result_dict.update({
            'ttr': ttr_value,
            'mtld': mtld_value,
            'hdd': hdd_value,

        })

        
    if vendi_score_computation:
        print("Computing Vendi Score...")

        # Compute the Vendi Score for all adaptations
        all_adaptations_joined = " ".join(all_adaptations)
        tokens = all_adaptations_joined.split()
        ngram_vs = text_utils.ngram_vendi_score(tokens, ns=[1, 2, 3])
        result_dict.update({
            'vendi_score_ngram_1_2': ngram_vs,
        })


    config_results.append(result_dict)

# === Save results ===
df_summary = pd.DataFrame(config_results)
df_summary.to_csv('res/adaptation/diversity_results.csv', index=False)


# %%
