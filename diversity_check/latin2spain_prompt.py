from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from ast import literal_eval
import re

# ---------- Constants and Configurations ----------

REPO_ID = "YOUR_REPO_ID"
FILENAME = "data.csv"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
batch_size = 3
k = 3  # Number of generations if sampling

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

# ---------- Recipe Helper Functions ----------

def extract_clean_recipe(text):
    match = re.findall(r'"""(Nombre:.*?Pasos:.*?)"""', text, re.DOTALL)
    if match:
        return match[-1].strip()

    match = re.search(r'(Nombre:.*?Pasos:.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()

def get_source_recipes(spanish=False):
    dataset = pd.read_csv(
        hf_hub_download(repo_id='somosnlp/RecetasDeLaAbuela', filename='recetasdelaabuela.csv', repo_type="dataset")
    )

    dataset = dataset[['Id','Nombre','Ingredientes','Pasos','Pais']]
    dataset['Ingredientes'] = dataset['Ingredientes'].str.replace('½','1/2')
    dataset['Pasos'] = dataset['Pasos'].str.replace('½','1/2')

    if spanish:
        df_cleaned = dataset[dataset['Pais'] == 'España'].dropna()
    else:
        df_cleaned = dataset[~dataset['Pais'].isin(['España', 'Internacional'])].dropna()

    dict_recipes = df_cleaned.to_dict('records')
    source_recipes = []

    for recipe in dict_recipes:
        ingredientes = ""
        if pd.notna(recipe['Ingredientes']):
            if isinstance(recipe['Ingredientes'], str):
                ingredientes = recipe['Ingredientes']
            else:
                for ing in literal_eval(recipe['Ingredientes']):
                    ingredientes += ing + "; "

        pasos = ""
        if pd.notna(recipe['Pasos']):
            if isinstance(recipe['Pasos'], str):
                pasos = recipe['Pasos']
            else:
                for paso in literal_eval(recipe['Pasos']):
                    pasos += paso + " "

        text_recipe = f"Nombre: {recipe['Nombre']}. Ingredientes: {ingredientes}. Pasos: {pasos}"
        source_recipes.append({'id': recipe['Id'], 'src': text_recipe, 'country': recipe['Pais']})

    return source_recipes

def chunkify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# ---------- Load Recipes and Model ----------

src_recipes = get_source_recipes()
spanish_recipes = get_source_recipes(spanish=True)

print(len(src_recipes), "Collected recipes not from Spain")
print(len(spanish_recipes), "Collected recipes from Spain")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ---------- Run Inference for Each Config ----------

for config in param_configs:
    temp = config["temperature"]
    top_k = config["top_k"]
    top_p = config["top_p"]

    results = []
    batch_number = 0
    use_sampling = temp > 0.0000000000001

    for batch in chunkify(src_recipes, batch_size):
        batch_number += 1
        print(f"🌡️ Batch {batch_number} - Temp: {temp} | top_k: {top_k} | top_p: {top_p}")

        prompts, ids, original_recipes, countries = [], [], [], []

        for recipe in batch:
            prompt = (
                f"Convierte la siguiente receta de {recipe['country']} en una receta española "
                f"para que encaje con la cultura española, sea consistente con el conocimiento alimenticio español, "
                f"y cumpla con el estilo de las recetas españolas y la disponibilidad de los alimentos. "
                f"""\"\"\"{recipe['src']}\"\"\" Formato obligatorio: proporciona solo la receta española con el formato de la receta original."""
            )
            prompts.append(prompt)
            ids.append(recipe['id'])
            original_recipes.append(recipe)
            countries.append(recipe['country'])

        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=use_sampling,
                temperature=temp,
                top_k=top_k if use_sampling else 0,
                top_p=top_p if use_sampling else 1.0,
                num_return_sequences=k if use_sampling else 1,
                pad_token_id=tokenizer.pad_token_id
            )

            batch_size_actual = len(batch)
            n_return = k if use_sampling else 1
            output_ids = output_ids.view(batch_size_actual, n_return, -1)

            for i in range(batch_size_actual):
                generations = []
                for j in range(n_return):
                    response = output_ids[i][j][inputs['input_ids'].shape[1]:]
                    raw_output = tokenizer.decode(output_ids[i][j], skip_special_tokens=True)
                    cleaned = extract_clean_recipe(raw_output)
                    generations.append(cleaned)

                results.append({
                    'id': ids[i],
                    'src': original_recipes[i],
                    'mod_list': generations,
                    'country': countries[i],
                    'adapted_to': 'Spain',
                    'config': {
                        'temperature': temp,
                        'top_k': top_k,
                        'top_p': top_p
                    }
                })

        except Exception as e:
            print("Error during generation:", str(e))
            for i in range(len(batch)):
                results.append({
                    'id': ids[i],
                    'src': original_recipes[i],
                    'mod_list': [f"error: {str(e)}"],
                    'country': countries[i],
                    'adapted_to': 'Spain',
                    'config': {
                        'temperature': temp,
                        'top_k': top_k,
                        'top_p': top_p
                    }
                })

        filename_partial = f'results/output_t{temp}_k{top_k}_p{top_p}_partial.json'
        with open(filename_partial, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    filename_complete = f'results/output_t{temp}_k{top_k}_p{top_p}.json'
    with open(filename_complete, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
