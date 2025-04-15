# File created by: andreamorgar
# 
# File with the following functions:
# - lexical_diversity
# - syntactic_diversity
# - semantic_diversity
# - compute_self_bleu
# - compute_lexical_diversity
# - compute_global_diversity
# - calculate_entropy
# - compute_local_diversity
#
# The functions are used to compute the diversity of the dataset.
# The functions are used in the notebook: diversity_check/compute_diversity.ipynb


from lexical_diversity import lex_div as ld
from collections import Counter
from nltk import ngrams, word_tokenize
import nltk
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from metrics import clean_ingredients, get_recipe_text
from lexical_diversity import lex_div as ld
from vendi_score import text_utils
from transformers import AutoModel, AutoTokenizer
import math
from collections import defaultdict
from nltk import word_tokenize

# Load Spanish model
nlp = spacy.load('es_core_news_sm')

# Download necessary NLTK data for Spanish
nltk.download('punkt', quiet=True)

def lexical_diversity(texts, n_values=[1, 2, 3]):
    total_unique_ngrams = 0
    total_ngrams = 0
    
    for n in n_values:
        all_ngrams = []
        
        for text in texts:
            tokens = word_tokenize(text.lower(), language='spanish')
            ngrams_list = list(ngrams(tokens, n))
            all_ngrams.extend(ngrams_list)
        
        ngrams_counts = Counter(all_ngrams)
        
        unique_ngrams = len(ngrams_counts)
        total_ngrams_in_all_texts = len(all_ngrams)
        
        total_unique_ngrams += unique_ngrams
        total_ngrams += total_ngrams_in_all_texts
    
    diversity_score = total_unique_ngrams / total_ngrams if total_ngrams > 0 else 0
    return diversity_score


# ----------- MSTTR -----------
def msttr(texts, segment_length=50, language='spanish'):
    all_ratios = []

    for text in texts:
        tokens = word_tokenize(text.lower(), language=language)
        if len(tokens) < segment_length:
            continue  # skip short texts
        
        # Split into fixed-length segments
        segments = [tokens[i:i+segment_length] for i in range(0, len(tokens), segment_length)]
        
        for segment in segments:
            types = set(segment)
            ratio = len(types) / len(segment)
            all_ratios.append(ratio)
    
    return np.mean(all_ratios) if all_ratios else 0


def syntactic_diversity(texts, n_values=[1, 2, 3]):
    total_unique_ngrams = 0
    total_ngrams = 0
    
    for n in n_values:
        all_pos_ngrams = []
        
        for text in texts:
            doc = nlp(text.lower())
            pos_tags = [token.pos_ for token in doc]
            pos_ngrams_list = list(ngrams(pos_tags, n))
            all_pos_ngrams.extend(pos_ngrams_list)
        
        pos_ngrams_counts = Counter(all_pos_ngrams)
        
        unique_ngrams = len(pos_ngrams_counts)
        total_ngrams_in_all_texts = len(all_pos_ngrams)
        
        total_unique_ngrams += unique_ngrams
        total_ngrams += total_ngrams_in_all_texts
    
    diversity_score = total_unique_ngrams / total_ngrams if total_ngrams > 0 else 0
    return diversity_score


# def semantic_diversity(texts, model_name='all-mpnet-base-v2'): #paraphrase-multilingual-MiniLM-L12-v2
#     model = SentenceTransformer(model_name)
    
#     # Encode the texts into sentence embeddings
#     embeddings = model.encode(texts)
    
#     # Print embeddings for debugging
#     # print("Embeddings shape:", embeddings.shape)
#     # print("Embeddings:", embeddings)
    
#     # Calculate the variance of the embeddings
#     variance = np.var(embeddings, axis=0).mean()
    
#     return variance




def semantic_diversity(texts, model_name='all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    
    similarity_matrix = cosine_similarity(embeddings)
    upper_triangular_values = similarity_matrix[np.triu_indices(len(texts), k=1)]
    mean_similarity = np.mean(upper_triangular_values)
    
    # Diversity is 1 - similarity (higher means more diverse)
    return 1 - mean_similarity




def compute_self_bleu(texts):
    """
    Computes the Self-BLEU score for a list of texts.
    
    Parameters:
        texts (list): A list of strings, where each string is a text sample.
    
    Returns:
        float: The average Self-BLEU score.
    """
    smoothie = SmoothingFunction().method4  # A smoothing function to avoid zero BLEU scores
    bleu_scores = []

    for i, text in enumerate(texts):
        references = texts[:i] + texts[i+1:]  # All other texts except the current one
        references = [nltk.word_tokenize(ref) for ref in references]  # Tokenize references
        hypothesis = nltk.word_tokenize(text)  # Tokenize the hypothesis
        score = sentence_bleu(references, hypothesis, smoothing_function=smoothie)
        bleu_scores.append(score)
    
    return sum(bleu_scores) / len(bleu_scores)




# all_ingredients = ingredient_count(results)
recipe_ingredients_cuisine ={}


def compute_lexical_diversity(df,separated_ingredients=True, tmp="original", ai_generated=False, mode = "ingredient_token", model_to_use="all-mpnet-base-v2"):

    results_analysis = []
    list_countries = list(set(list(df["country"])))
    country_count = []

    bert_vs = 0
    roberta_vs = 0

    for country in list_countries:
        # print("Computing ", country)
        country_data = df[df['country'] == country]


        if mode == "ingredient_token": # this mode is for preprocessed ingredient lists. 
            final_ingredients_ = clean_ingredients(country_data.to_dict('records'), ai_generated=ai_generated)
            if separated_ingredients:
                recipe_ingredients_cuisine[country] = [i['Ingredientes_pr'] for i in final_ingredients_]
                ing_tokens = []
                # join each item
                for i in recipe_ingredients_cuisine[country]:
                    # join in a space the elements in the list i
                    ing_tokens.append(" ".join(i))
                ngram_vs = text_utils.ngram_vendi_score(ing_tokens, ns=[1, 2])

                lexical_ngram_score = lexical_diversity(ing_tokens)
                syntactic_score = syntactic_diversity(ing_tokens)
                semantic_score = semantic_diversity(ing_tokens, model_to_use)
                
                # bert_vs = text_utils.embedding_vendi_score(only_texts, model_path="dccuchile/bert-base-spanish-wwm-cased")
                # roberta_vs = text_utils.embedding_vendi_score(only_texts, model_path="bertin-project/bertin-roberta-base-spanish")

                recipe_ingredients_cuisine[country] = [item for sublist in recipe_ingredients_cuisine[country] for item in sublist]
            else:
                recipe_ingredients_cuisine[country] = [i['Ingredientes_pr'] for i in final_ingredients_]
                recipe_ingredients_cuisine[country] = [item for sublist in recipe_ingredients_cuisine[country] for item in sublist]
                # create a list when each word is an element of the list
                recipe_ingredients_cuisine[country] = " ".join(recipe_ingredients_cuisine[country])

    

        if mode == "whole_recipe":
            recipe_texts  = get_recipe_text(country_data.to_dict('records'), ai_generated=ai_generated) 
            
            only_texts = [i['ingredient_text'] for i in recipe_texts]
            ngram_vs = text_utils.ngram_vendi_score(only_texts, ns=[1, 2])
            lexical_ngram_score = lexical_diversity(only_texts)
            syntactic_score = syntactic_diversity(only_texts)
            semantic_score = semantic_diversity(only_texts)
            
            # bert_vs = text_utils.embedding_vendi_score(only_texts, model_path="dccuchile/bert-base-spanish-wwm-cased")
            # roberta_vs = text_utils.embedding_vendi_score(only_texts, model_path="bertin-project/bertin-roberta-base-spanish")
            
            # tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
            # self_bleu = text_utils.self_bleu(only_texts,tokenizer)
            recipe_ingredients_cuisine[country] = []
            for i in recipe_texts:
                recipe_ingredients_cuisine[country].extend(ld.tokenize(i['recipe_text']))

        if mode == "ingredient_text":
            ing_texts =  get_ingredient_text(country_data.to_dict('records'), ai_generated=ai_generated)   
            
            only_texts = [i['ingredient_text'] for i in ing_texts]
            ngram_vs = text_utils.ngram_vendi_score(only_texts, ns=[1, 2])

            lexical_ngram_score = lexical_diversity(only_texts)
            syntactic_score = syntactic_diversity(only_texts)
            semantic_score = semantic_diversity(only_texts)
            # bert_vs = text_utils.embedding_vendi_score(only_texts, model_path="dccuchile/bert-base-spanish-wwm-cased")
            # roberta_vs = text_utils.embedding_vendi_score(only_texts, model_path="bertin-project/bertin-roberta-base-spanish")

            recipe_ingredients_cuisine[country] = []
            for i in ing_texts:
                recipe_ingredients_cuisine[country].extend(ld.tokenize(i['ingredient_text']))

        
        # print("Computing lexical diversity")
        ttr_value = ld.ttr(recipe_ingredients_cuisine[country])
        mtld_value = ld.mtld(recipe_ingredients_cuisine[country])
        vocd_value = ld.hdd(recipe_ingredients_cuisine[country]) 

        print(f"N-grams: {ngram_vs:.02f}, BERT: {bert_vs:.02f}, ROBERTA: {roberta_vs:.02f}, lexical_diversity: {lexical_ngram_score:.02f}, syntactic_diversity: {syntactic_score:.02f}, semantic_diversity: {semantic_score:.02f}")  

        results_analysis.append({"country": country, "ttr": ttr_value, "mtld": mtld_value, "vocd": vocd_value, "ngram_vs": ngram_vs, 
                                 "bert_vs":bert_vs, "roberta_vs":roberta_vs, "lexical_ngram":lexical_ngram_score, 
                                 "syntatic_score":syntactic_score, "semantic_score":semantic_score, "mode": tmp, "type":mode}) #, "separated_ingredients":separated_ingredients})
        country_data

    # reorder to have country and separated_ingredients as columns in dataframe
    df = pd.DataFrame(results_analysis)

    df = df.pivot(index='country', columns='mode', values=['ttr', 'mtld', 'vocd', 'ngram_vs', 'bert_vs', 'roberta_vs', 'lexical_ngram', 'syntatic_score', 'semantic_score'])
    df = df.reset_index()
    df.columns.name = None
    df.to_latex("latex_tables/lexical_diversity_"+ mode +".tex",index=False)
    return results_analysis

def compute_global_diversity_from_column(df, column='Ingredientes_pr', mode="original"):
    results_analysis = []
    list_countries = df["country"].unique().tolist()
    print("Computing global diversity from column", column, "for mode", mode)
    
    for country in list_countries:
        country_data = df[df['country'] == country]

        ingredient_list = []
        for item in country_data[column]:
            if isinstance(item, list):
                if all(isinstance(subitem, list) for subitem in item):
                    # Lista de listas (k generaciones)
                    for sublist in item:
                        ingredient_list.extend(sublist)
                else:
                    # Lista plana de ingredientes
                    ingredient_list.extend(item)

        global_diversity = len(set(ingredient_list))
        results_analysis.append({
            "country": country,
            "mode": mode,
            "ingredients": ingredient_list,
            "global": global_diversity
        })

    return results_analysis


def compute_global_diversity(df, tmp="original", ai_generated=False):
    results_analysis = []
    list_countries = list(set(list(df["country"])))
    print("Computing global diversity for mode", tmp)   
    for country in list_countries:
        # print("Computing ", country)
        country_data = df[df['country'] == country]
        # get average length of Ingredients
        final_ingredients_ = clean_ingredients(country_data.to_dict('records'), ai_generated=ai_generated)

        ingredient_list = []
        for i in final_ingredients_:
            ingredient_list.extend(i['Ingredientes_pr'])
    
        global_diversity = len(set(ingredient_list))
        results_analysis.append({"country": country, "mode": tmp, "ingredients":ingredient_list, "global":global_diversity}) 
    
    return results_analysis

def calculate_entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:  # Avoid log(0)
            entropy -= p * math.log(p)
    return entropy

def compute_local_diversity(df):
    df['country_mode'] = df['country'] + '_' + df['mode'].astype(str)
    total_entropies = []
    list_tmp = [0, '0.3', '0.6', '0.9']
    for tmp in list_tmp:
        # filter by temperature balanced_dataset
        df_tmp = df[(df['mode'] == tmp) | (df['mode'] == -1)]
        print("Computing local diversity for mode", tmp)

        all_ingredients = set()

        for element in df_tmp.to_dict('records'):
            #for dish in element:
            all_ingredients.update(element['ingredients'])

        all_ingredients = sorted(all_ingredients)  # Sort to ensure consistent order
        ingredient_index = {ingredient: idx for idx, ingredient in enumerate(all_ingredients)}

        # Step 2: Count occurrences of each ingredient in each cuisine
        cuisine_ingredient_counts = defaultdict(lambda: defaultdict(int))

        for country in df_tmp.to_dict('records'):
            for ingredient in country['ingredients']:
                cuisine_ingredient_counts[country['country_mode']][ingredient] += 1

        # Step 3: Create and normalize vectors for each cuisine
        cuisine_vectors = {}

        for cuisine, ingredient_counts in cuisine_ingredient_counts.items():
            vector = [0] * len(all_ingredients)
            total_ingredients = sum(ingredient_counts.values())
            for ingredient, count in ingredient_counts.items():
                index = ingredient_index[ingredient]
                vector[index] = count / total_ingredients  # Normalize by the total count of ingredients in the cuisine
            cuisine_vectors[cuisine] = vector

        # Step 4: Calculate entropy for each cuisine
        cuisine_entropies = {}

        for cuisine, vector in cuisine_vectors.items():
            entropy = calculate_entropy(vector)
            cuisine_entropies[cuisine] = entropy
                #results_analysis = compute_ingredients(df_tmp,tmp=tmp, ai_generated=True) 
                # concat to df
                #df = pd.concat([df, pd.DataFrame(results_analysis)], ignore_index=True)

            total_entropies.append({'country_': cuisine, 'entropy': entropy})
    return total_entropies


# modular version of compute_local_diversity 
def compute_local_diversity_from_column(df, column='Ingredientes_pr', mode_label=''):
    df['country_mode'] = df['country'] + '_' + df['mode'].astype(str)
    total_entropies = []

    all_ingredients = set()
    for ingredients in df[column]:
        if isinstance(ingredients, list):
            if all(isinstance(sublist, list) for sublist in ingredients):
                for sublist in ingredients:
                    all_ingredients.update(sublist)
            else:
                all_ingredients.update(ingredients)

    all_ingredients = sorted(all_ingredients)
    ingredient_index = {ingredient: idx for idx, ingredient in enumerate(all_ingredients)}

    # Count occurrences of each ingredient per cuisine
    cuisine_ingredient_counts = defaultdict(lambda: defaultdict(int))

    for row in df.to_dict('records'):
        ingredients = row[column]
        if isinstance(ingredients, list) and all(isinstance(sublist, list) for sublist in ingredients):
            flattened = [ing for sub in ingredients for ing in sub]
        else:
            flattened = ingredients if isinstance(ingredients, list) else []

        for ingredient in flattened:
            cuisine_ingredient_counts[row['country_mode']][ingredient] += 1

    # Create and normalize vectors
    cuisine_vectors = {}
    for cuisine, ingredient_counts in cuisine_ingredient_counts.items():
        vector = [0] * len(all_ingredients)
        total = sum(ingredient_counts.values())
        for ing, count in ingredient_counts.items():
            idx = ingredient_index[ing]
            vector[idx] = count / total
        cuisine_vectors[cuisine] = vector

    # Compute entropy per cuisine
    for cuisine, vector in cuisine_vectors.items():
        entropy = calculate_entropy(vector)
        total_entropies.append({
            'country_mode': cuisine,
            'entropy': entropy,
            'mode': mode_label,
        })

    return total_entropies
