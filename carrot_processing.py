from datasets import load_dataset
import pandas as pd
def load_data(filter_county=""):
    ds = load_dataset("somosnlp/RecetasDeLaAbuela", "version_1")

    recipe_titles = []
    recipes_map = {}

    for item in ds['train']:
        nombre = item['Nombre']
        ingredientes = item['Ingredientes']
        pasos = item['Pasos']
        ingredientes = ingredientes.replace("\n"," ").replace("\t"," ")
        pasos = pasos.replace("\n"," ").replace("\t"," ")
        if filter_county != "" and item['Pais'] == filter_county:
            recipe_titles.append(nombre)
            recipes_map[nombre] = (ingredientes,pasos)
    return recipe_titles, recipes_map

def recipe_split(data):
    data = data.split('\t')
    title = data[0]
    if len(data) >= 2:
        ingredients = data[1]
    else:
        ingredients = ""
    
    if len(data) >= 2:
        steps = data[2]
    else:
        steps = ""

    return title, ingredients, steps 

def generate_recipe_adaption_query():
    ds = load_dataset("somosnlp/RecetasDeLaAbuela", "version_1")
   
    recipes = []

    for item in ds['train']:
        nombre = item['Nombre']
        ingredientes = item['Ingredientes']
        pasos = item['Pasos']
        ingredientes = ingredientes.replace("\n"," ").replace("\t"," ")
        pasos = pasos.replace("\n"," ").replace("\t"," ")
        if item['Pais'] != 'ESP' and item['Pais'] != None:
            recipes.append(nombre + '\t' + ingredientes + "\t" + pasos)

    with open("data/all_recipes.txt", "w", encoding="utf-8") as f:
        for recipe in recipes:
            f.write(recipe + "\n")

def load_recipe_adaption_query(dir):
    queries = []
    with open(dir, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(line.strip())
    return queries