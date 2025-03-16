import inflect
import re
from unidecode import unidecode
import ast
import pandas as pd

# Initialize the inflect engine
p = inflect.engine()

# Función para limpiar las especificaciones
def limpiar_especificaciones(ingredientes, palabras_eliminar):
    ingredientes_limpios = []
    for ingrediente in ingredientes:
        for palabra in palabras_eliminar:
            ingrediente = ingrediente.replace(palabra, "").strip()
        ingredientes_limpios.append(ingrediente)
    return ingredientes_limpios

def remove_empty_strings(lst):
    return [item for item in lst if item != '']

def remove_text_in_parentheses(text):
    # This regex matches any text between parentheses and the parentheses themselves
    result = re.sub(r'\(.*?\)', '', text)
    return result

def remove_text_after_coma(text):
    # remove text after coma
    result = text.split(',')[0]   
    return result

def remove_punctuation(text):
    # Remove punctuation using regex
    return re.sub(r'[^\w\s]', '', text)

def singularize_ingredient(ingredient):
    # Split the ingredient into words
    words = ingredient.split()
    # Singularize each word if it is plural
    singularized_words = [p.singular_noun(word) if p.singular_noun(word) else word for word in words]
    # Join the words back into a single string
    singularized_ingredient = ' '.join(singularized_words)
    return singularized_ingredient

def remove_everything_after(list_words, text):
    for word in list_words:
        if word in text:
            text = text.split(word)[0]
    return text


# Función para limpiar cada línea del texto
def limpiar_linea(linea,pattern):
    return re.sub(pattern, '', linea, flags=re.IGNORECASE).strip()

def clean_ingredients(recipes_dict,ai_generated=False):

    remove_after_word = [' entre', ' al gusto', ' para', 'para ']
    palabras_eliminar = ["mediana", "bastantes", "bastante", "rebanadas de", "pequeña", "trozos medianos", 
                     "al gusto", "a gusto", "trozos de ", "trozo de ", "en trozos pequeños", ' ya ',
                       " ()", "pequeño", "regulares", "lb. ", " suficiente"]
    for i,elem in enumerate(recipes_dict):

        if ai_generated:
            texto = (elem['mod'].split("Ingredientes")[1]).split("\n\n")[1].replace(" ", " ")
            
            texto = texto.lower() # lowercase
            texto = texto.replace("¼", "1/4")
            texto = texto.replace("½", "1/2")
            texto = texto.replace("¾", "3/4")
            texto = texto.replace("⅓", "1/3")
            texto = texto.replace("Al gusto,", "")
            texto = texto.replace("Al gusto:", "")


            ingredientes = texto.split("* ")
            ingredientes_principal = [i.split(',')[0] for i in ingredientes]
            ingredientes_principal = [i.split('(')[0] for i in ingredientes_principal]

        else:

            if pd.isna(elem['Ingredientes']):
                elem['Ingredientes_pr'] = ''
                continue

            if elem['Ingredientes'].strip().startswith("["): #some cases they save the ingredients as a list
                # Safely evaluate the string to convert it to a list
                ingredientes = ast.literal_eval(elem['Ingredientes'])
                # lowercase
                ingredientes = [ing.lower() for ing in ingredientes]

                ingredientes_principal = [remove_text_in_parentheses(ing) for ing in ingredientes]
                ingredientes_principal = [remove_text_after_coma(ing) for ing in ingredientes_principal]
                ingredientes_principal = [remove_everything_after(remove_after_word, ing) for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("¼", "1/4") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("½", "1/2") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("¾", "3/4") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("⅓", "1/3") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("cta.de,", "") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("cinco:", "5") for ing in ingredientes_principal]
                ingredientes_principal = [ing.replace("azãºcar", "azúcar") for ing in ingredientes_principal]



            else:
                texto = remove_text_in_parentheses(elem['Ingredientes'])
                texto = remove_text_after_coma(texto)
                texto = remove_everything_after(remove_after_word, texto)
                # texto = remove_punctuation(texto) arreglar porque hay datos que tienen signos de puntuacion---------------------------todo
                # lowercase
                texto = texto.lower()
                texto = texto.replace("¼", "1/4")
                texto = texto.replace("½", "1/2")
                texto = texto.replace("¾", "3/4")
                texto = texto.replace("⅓", "1/3")
                texto = texto.replace("al gusto,", "")
                texto = texto.replace("al gusto:", "")
                texto = texto.replace("cta.de", "")
                texto = texto.replace("cinco", "5")
                texto = texto.replace("azãºcar", "azúcar")
    

                ingredientes_principal = texto.split(',')


        pattern = r'\b\d+\s*(?:\/\d+)?\s*(?:g|gramos|pieza|centímetro cúbico|cucharada sopera|cc|un kilo|tableta|pisca|piezas|paquete|paquetes|un kilo|tarro|loncha|cabeza|cabezas|lonchas|kilogramo|kilogramos|puñados|chorrito|botella|gr.|libra|ralladura|lámina|barra|paquete|bastante|caja|rama|puñado|manojo|bote|vaso|pellizco|unidad|chorro|vaso|lata|rama|postre|litro|litros|mililitros|barra|-|cucharada colmada|unidades|copa|un kilo|kilo|gr|kg|ml|cl|dl|l|cm|bolas|dientes|diente|pizca|cucharadas soperas|tiras|tajadas|cucharaditas|cucharadita|cucharadas|mix|cucharada|cc|cda|cdta)?(?:\s+(?!de\b)(?:taza|tazas))?\b\s*(?:de)?'

        # print(ingredientes_principal, 'ingredientes principal')
        # Aplicar la limpieza a cada línea y eliminar líneas vacías
        lineas_limpias = []
        lineas_limpias = [limpiar_linea(linea,pattern).lower() for linea in ingredientes_principal if linea.strip()]

        # Eliminar la palabra "taza" o "tazas" si aparece al inicio de un elemento limpio
        lineas_limpias_final = [re.sub(r'^(taza de|tazas de)\s+', '', linea, flags=re.IGNORECASE) for linea in lineas_limpias]

        final_ingredients = []
        for i in lineas_limpias_final:
            if ' y ' in i:
                two_ingredientes = i.split(' y ')
                final_ingredients.append(two_ingredientes[0])
                # final_ingredients.append(two_ingredientes[1])
            elif 'o' in i:
                two_ingredientes = i.split(' o ')
                final_ingredients.append(two_ingredientes[0])
            else:
                final_ingredients.append(i)


        elem['Ingredientes_pr'] = limpiar_especificaciones(final_ingredients, palabras_eliminar)
        elem['Ingredientes_pr'] = [singularize_ingredient(ingredient).rstrip('.') for ingredient in elem['Ingredientes_pr'] ]


        elem['Ingredientes_pr'] = [limpiar_linea(linea,pattern).lower().rstrip('.') for linea in elem['Ingredientes_pr']]
        elem['Ingredientes_pr'] = remove_empty_strings(elem["Ingredientes_pr"])



        elem['Ingredientes_pr'] = limpiar_especificaciones(elem['Ingredientes_pr'], ['cc de ','. de ', '. ', '- ', 'c/n ', ' c/n', 'ajã\xade', 'ajã\xad ', 'ðÿ§…', ' %', 'taza ', 'centimetro cubico de ', 'c.c', 'una ', 'trozo de ', 'un kilo de ', ' en cuadradito', 'semilla de ', 'copita de '])
        # unidecode
        elem['Ingredientes_pr'] = [unidecode(ing) for ing in elem['Ingredientes_pr']]

    return recipes_dict


def get_ingredient_text(recipes_dict,ai_generated=False):

    # This function extracts the ingredient text from the recipes
    """ 
    Args:
    recipes_dict: list of dictionaries with the recipes
    ai_generated: boolean to indicate if the recipes are AI generated or not
    Returns:
    recipes_dict: list of dictionaries with the recipes and the ingredient text
    """ 

    for i,elem in enumerate(recipes_dict):
        if ai_generated:
            final_text = (elem['mod'].split("Ingredientes")[1]).split("\n\n")[1].replace(" ", " ")
            
        else:
            if pd.isna(elem['Ingredientes']):
                final_text = ''

            elif elem['Ingredientes'].strip().startswith("["): #some cases they save the ingredients as a list
                # Safely evaluate the string to convert it to a list
                ingredientes = ast.literal_eval(elem['Ingredientes'])
                # lowercase
                # join elements of the list in a text
                final_text = ', '.join(ingredientes)  
            else:
                final_text = elem['Ingredientes']
                
        final_text = final_text.lower()
        elem['ingredient_text'] = unidecode(final_text)
    return recipes_dict


def get_recipe_text(recipes_dict,ai_generated=False):
    # This function extracts the recipe text from the recipes
    """
    Args:
    recipes_dict: list of dictionaries with the recipes
    ai_generated: boolean to indicate if the recipes are AI generated or not
    Returns:
    recipes_dict: list of dictionaries with the recipes and the recipe text
    """

    ingredients = get_ingredient_text(recipes_dict,ai_generated)

    for i,elem in enumerate(recipes_dict):
        
        if ai_generated:
            final_text = elem['mod']
            
        else:
            if pd.isna(elem['Pasos']):
                final_pasos = ''    
            elif elem['Pasos'].strip().startswith("["): #some cases they save the ingredients as a list
                
                # Safely evaluate the string to convert it to a list
                try:
                    pasos = ast.literal_eval(elem['Pasos'])
                except:
                    pasos = ast.literal_eval(elem['Pasos']+ "']")

                final_pasos = ', '.join(pasos)  
            else:
                final_pasos = elem['Pasos']
                
                
            final_text = "Título: " + elem['Nombre'] + "\n" + ". Ingredientes: " + ingredients[i]["ingredient_text"] + "\n" + ". Pasos: " + final_pasos
        # print(final_text)
        try:
            elem['recipe_text'] = unidecode(final_text.encode('latin1').decode('utf-8').lower()) # fix wrong encoding
        except UnicodeDecodeError: # avoid for some cases that are already in utf-8
            elem['recipe_text'] = unidecode(final_text.lower())
        except UnicodeEncodeError: # avoid for some cases that are already in utf-8
            elem['recipe_text'] = unidecode(final_text.lower())
    return recipes_dict

# to test
# results = get_recipe_text(spanish_recipes.to_dict('records'))
#results = get_recipe_text(spanish_recipes.to_dict('records'))
# results = get_recipe_text(original_recipes.to_dict('records'))
