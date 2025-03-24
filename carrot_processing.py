from datasets import load_dataset

def load_data(filter_county=""):
    ds = load_dataset("somosnlp/RecetasDeLaAbuela", "version_1")

    recipe_titles = []
    recipes_map = {}

    for item in ds['train']:
        nombre = item['Nombre']
        ingredientes = item['Ingredientes']
        pasos = item['Pasos']

        if filter_county != "" and item['Pais'] == filter_county:
            recipe_titles.append(nombre)
            recipes_map[nombre] = "Nombre: " + nombre + '\nIngredientes: ' + ingredientes + "\nPasos: " + pasos
    return recipe_titles, recipes_map

def recipe_split(data):
    data = data.split('\t')
    title = " ".join(data[0].split(" ")[1:])
    if len(data) >= 3:
        content = data[1] + data[2]
    else:
        content = data[1]
        
    return title, content 

def generate_recipe_adaption_query():
    ds = load_dataset("somosnlp/RecetasDeLaAbuela", "version_1")
   
    recipes = []

    for item in ds['train']:
        nombre = item['Nombre']
        ingredientes = item['Ingredientes']
        pasos = item['Pasos']
        ingredientes = ingredientes.replace("\n"," ")
        pasos = pasos.replace("\n"," ")
        if item['Pais'] != 'ESP' and item['Pais'] != None:
            recipes.append("Nombre: " + nombre + '\tIngredientes: ' + ingredientes + "\tPasos: " + pasos)

    with open("data/all_recipes.txt", "w", encoding="utf-8") as f:
        for recipe in recipes:
            f.write(recipe + "\n")

def load_recipe_adaption_query(dir):
    queries = []
    with open(dir, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(line.strip())
    return queries

generate_recipe_adaption_query()