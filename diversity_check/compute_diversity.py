# File to compute for the recipe datasets the metrics defined in diversity_metrics.py
import pandas as pd
from diversity_metrics import compute_lexical_diversity, compute_global_diversity, compute_local_diversity
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings



def lexical_diversity(original, adapted):
    final_df = pd.DataFrame()

    print("Computing original lexical diversity")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_analysis = compute_lexical_diversity(original,separated_ingredients=True, mode="whole_recipe") #one token one ingredient. Here we consider each ingredient as a token in the list. This makes the diversity higher.
    results_analysis
    
    #reorder to have country and separated_ingredients as columns in dataframe
    df = pd.DataFrame(results_analysis)


    print("Computing adapted lexical diversity")
    list_tmp = ['1e-13', '0.3', '0.6', '0.9']
    for tmp in list_tmp:
        print(" --- Computing temperature value",tmp)
        # filter by temperature balanced_dataset
        df_tmp = adapted[adapted['temperature'] == tmp]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results_analysis = compute_lexical_diversity(df_tmp,separated_ingredients=True, tmp=tmp, ai_generated=True, mode="whole_recipe")
        
        # concat to df
        df = pd.concat([df, pd.DataFrame(results_analysis)], ignore_index=True)

    df['mode'] = df['mode'].replace('original', -1)
    df['mode'] = df['mode'].replace('1e-13', 0)

    final_df = pd.concat([final_df, df], ignore_index=True)

    final_df 

    final_df.to_csv("res/lexical_diversity_results_all.csv", index=False)

    return final_df



def global_diversity(original, adapted):
    final_df = pd.DataFrame()

    print("Computing original global diversity")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_analysis = compute_global_diversity(original,tmp="original", ai_generated=False) #one token one word
    results_analysis
    
    #reorder to have country and separated_ingredients as columns in dataframe
    df = pd.DataFrame(results_analysis)

    print("Computing adapted global diversity")
    list_tmp = ['1e-13', '0.3', '0.6', '0.9']
    for tmp in list_tmp:
        # filter by temperature balanced_dataset
        df_tmp = adapted[adapted['temperature'] == tmp]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results_analysis = compute_global_diversity(df_tmp,tmp=tmp, ai_generated=True) 
        
        # concat to df
        df = pd.concat([df, pd.DataFrame(results_analysis)], ignore_index=True)

    df['mode'] = df['mode'].replace('original', -1)
    df['mode'] = df['mode'].replace('1e-13', 0)
    df_with_ingredients = df.copy()
    df['global'] = df['global'].astype(int)
    df_diversity = df.pivot(index='country', columns='mode', values=['global'])
    df_diversity = df_diversity.reset_index()
    df_diversity.columns.name = None
    df_diversity.to_latex("latex_tables/global_diversity.tex",index=False)
    return df_diversity, df_with_ingredients


# Compute local diversity
def local_diversity(df):
    total_entropies = compute_local_diversity(df)
    df_local_div = pd.DataFrame(total_entropies)
    df_local_div.to_csv("res/local_diversity_results_all.csv", index=False)

    # remove duplicates
    df_local_div[['country', 'mode']] = df_local_div['country_'].str.split('_', expand=True)
    df_local_div = df_local_div.drop(columns=['country_'])
    df_local_div = df_local_div.drop_duplicates()
    df_local_div.head(50)

    # # Pivot the DataFrame
    df_pivoted = df_local_div.pivot_table(index='country', columns='mode', values=['entropy'])
    # Reset the index to make 'country' a column again
    df_pivoted.reset_index()

    # Rename columns for clarity (optional)
    df_pivoted.columns.name = None

    df_pivoted.to_latex("latex_tables/local_diversity.tex")
    return df_pivoted




balanced_adapted_dataset = pd.read_csv("res/balanced_adapted_recipes.csv")
balanced_original_dataset = pd.read_csv("res/balanced_original_recipes.csv")
balanced_adapted_dataset['temperature'] = balanced_adapted_dataset['temperature'].astype(str)

df_lexical_diversity = lexical_diversity(balanced_original_dataset, balanced_adapted_dataset)
df_global_diversity, df_ingredients = global_diversity(balanced_original_dataset, balanced_adapted_dataset)
df_local_diversity = local_diversity(df_ingredients)
