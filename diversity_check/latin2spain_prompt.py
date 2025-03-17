from datasets import load_dataset

from huggingface_hub import hf_hub_download
import pandas as pd
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

REPO_ID = "YOUR_REPO_ID"
FILENAME = "data.csv"

from ast import literal_eval


def get_source_recipes(spanish=False):
    dataset = pd.read_csv(
        hf_hub_download(repo_id='somosnlp/RecetasDeLaAbuela', filename='recetasdelaabuela.csv', repo_type="dataset")
    )

    dataset = dataset[['Id','Nombre','Ingredientes','Pasos','Pais']]
    dataset['Ingredientes'] = dataset['Ingredientes'].str.replace('½','1/2')
    dataset['Pasos'] = dataset['Pasos'].str.replace('½','1/2')

    if spanish:
        #get recetas from Spain
        spanish_recipes = dataset[dataset['Pais'] == 'España']
        df_cleaned = spanish_recipes.dropna()
         
    else:
        # get rest of the recipes (the country is Spain, neither is International at the moment)
        df = dataset[~dataset['Pais'].isin(['España', 'Internacional'])]
        df_cleaned = df.dropna()

    dict_recipes = df_cleaned.to_dict('records')
    source_recipes = []

    for i,recipe in enumerate(dict_recipes):

        ingredientes = ""
        if recipe['Ingredientes'] is not np.nan:
            if isinstance(recipe['Ingredientes'], str):
                ingredientes = recipe['Ingredientes']
            else:
                for ing in literal_eval(recipe['Ingredientes']):
                    ingredientes +=ing 
                    ingredientes+="; "

        pasos = ""
        if recipe['Pasos'] is not np.nan:
            if isinstance(recipe['Pasos'], str):
                pasos = recipe['Pasos']

            else:
                for paso in literal_eval(recipe['Pasos']):
                    pasos +=paso 
                    pasos+=" "

        text_recipe = "Nombre: "+recipe['Nombre'] +". Ingredientes: " + ingredientes + ". Pasos: " + pasos 
        source_recipes.append({'id':recipe['Id'], 'src':text_recipe, 'country':recipe['Pais']}) 
    return source_recipes

src_recipes = get_source_recipes()
spanish_recipes = get_source_recipes(spanish=True)

print(len(src_recipes), "Collected recipes not from Spain")
print(len(spanish_recipes), "Collected recipes from Spain")


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

temperature_list = [0.0000000000001, 0.3, 0.6, 0.9]
for temp in temperature_list:
    results = [] 
    cnt = 0
    for recipe in src_recipes:
        cnt += 1
        try:


            msg1 = "Convierte la siguiente receta de "+ recipe['country'] +" en una receta española para que encaje con la cultura española, sea consistente con el conocimiento alimenticio español, y cumpla con el estilo de las recetas españolas y la disponibilidad de los alimentos. """ + recipe['src'] + "Formato obligatorio: proporciona solo la receta española con el formato de la receta original." 



            messages = [
                {"role": "user", "content": msg1}
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            text_response= tokenizer.decode(response, skip_special_tokens=True)

            
            results.append({'id':recipe['id'], 'src':recipe, 'mod':text_response, 'country':recipe['country']}) 
        except:
            results.append({'id':recipe['id'], 'src':recipe, 'mod':"error", 'country':recipe['country']})


        if cnt % 10 == 0:
            with open(f'results/output_{temp}.json', 'w' , encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False)

