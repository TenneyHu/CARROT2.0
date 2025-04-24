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



# def clean_ingredients(recipes_dict, ai_generated=False, name_column='mod'):
#     remove_after_word = [' entre', ' al gusto', ' para', 'para ']
#     palabras_eliminar = [
#         "mediana", "bastantes", "bastante", "rebanadas de", "pequeña", "trozos medianos",
#         "al gusto", "a gusto", "trozos de ", "trozo de ", "en trozos pequeños", ' ya ',
#         " ()", "pequeño", "regulares", "lb. ", " suficiente"
#     ]

#     for i, elem in enumerate(recipes_dict):
#         print(f"Procesando receta {i}")

#         if ai_generated:
#             texto = elem[name_column].split("Ingredientes", 1)[-1]
#             texto = texto.split("Pasos:", 1)[0].strip()
#             texto = texto.lstrip(": ").strip().lower()

#             texto = texto.replace("¼", "1/4").replace("½", "1/2").replace("¾", "3/4").replace("⅓", "1/3")
#             texto = texto.replace("al gusto,", "").replace("al gusto:", "")

#             # Detectar separador adecuado
#             if "* " in texto:
#                 ingredientes = texto.split("* ")
#             elif "." in texto:
#                 ingredientes = texto.split(".")
#             else:
#                 ingredientes = texto.split(",")

#             # Limpieza cuidadosa
#             ingredientes_principal = []
#             for ing in ingredientes:
#                 ing = ing.strip()
#                 if not ing:
#                     continue
#                 # if " sin " in ing:
#                 #     ing = ing.split(" sin ")[0].strip()
#                 # Solo eliminar lo que viene después de " sin " si no hay número o unidad detrás
#                 if re.search(r"\b sin \b", ing) and not re.search(r"\d", ing):
#                     ing = ing.split(" sin ")[0].strip()
#                 ing = re.sub(r"\([^)]*\)", "", ing).strip()
#                 ingredientes_principal.append(ing)

#         else:
#             if pd.isna(elem['Ingredientes']):
#                 elem['Ingredientes_pr'] = ''
#                 continue

#             if elem['Ingredientes'].strip().startswith("["):
#                 ingredientes = ast.literal_eval(elem['Ingredientes'])
#                 ingredientes = [ing.lower() for ing in ingredientes]
#                 ingredientes_principal = [remove_text_in_parentheses(ing) for ing in ingredientes]
#                 ingredientes_principal = [remove_text_after_coma(ing) for ing in ingredientes_principal]
#                 ingredientes_principal = [remove_everything_after(remove_after_word, ing) for ing in ingredientes_principal]
#                 ingredientes_principal = [ing.replace("¼", "1/4") for ing in ingredientes_principal]
#                 ingredientes_principal = [ing.replace("½", "1/2") for ing in ingredientes_principal]
#                 ingredientes_principal = [ing.replace("¾", "3/4") for ing in ingredientes_principal]
#                 ingredientes_principal = [ing.replace("⅓", "1/3") for ing in ingredientes_principal]
#                 ingredientes_principal = [ing.replace("cta.de,", "") for ing in ingredientes_principal]
#                 ingredientes_principal = [ing.replace("cinco:", "5") for ing in ingredientes_principal]
#                 ingredientes_principal = [ing.replace("azãºcar", "azúcar") for ing in ingredientes_principal]
#             else:
#                 texto = remove_text_in_parentheses(elem['Ingredientes'])
#                 texto = remove_text_after_coma(texto)
#                 texto = remove_everything_after(remove_after_word, texto)
#                 texto = texto.lower()
#                 texto = texto.replace("¼", "1/4").replace("½", "1/2").replace("¾", "3/4").replace("⅓", "1/3")
#                 texto = texto.replace("al gusto,", "").replace("al gusto:", "")
#                 texto = texto.replace("cta.de", "").replace("cinco", "5").replace("azãºcar", "azúcar")
#                 ingredientes_principal = texto.split(',')

#         # Limpieza común
#         pattern = r'\b\d+\s*(?:\/\d+)?\s*(?:g|gramos|pieza|centímetro cúbico|cucharada sopera|cc|un kilo|tableta|pisca|piezas|paquete|paquetes|tarro|loncha|cabeza|cabezas|lonchas|kilogramo|kilogramos|puñados|chorrito|botella|gr.|libra|ralladura|lámina|barra|caja|rama|puñado|manojo|bote|vaso|pellizco|unidad|chorro|lata|postre|litro|litros|mililitros|barra|-|cucharada colmada|unidades|copa|kilo|gr|kg|ml|cl|dl|l|cm|bolas|dientes|diente|pizca|cucharadas soperas|tiras|tajadas|cucharaditas|cucharadita|cucharadas|mix|cucharada|cc|cda|cdta)?(?:\s+(?!de\b)(?:taza|tazas))?\b\s*(?:de)?'

#         lineas_limpias = [limpiar_linea(linea, pattern).lower() for linea in ingredientes_principal if linea.strip()]
#         lineas_limpias_final = [re.sub(r'^(taza de|tazas de)\s+', '', linea, flags=re.IGNORECASE) for linea in lineas_limpias]

#         # final_ingredients = []
#         # for linea in lineas_limpias_final:
#         #     if ' y ' in linea:
#         #         final_ingredients.append(linea.split(' y ')[0])
#         #     elif ' o ' in linea:
#         #         final_ingredients.append(linea.split(' o ')[0])
#         #     else:
#         #         final_ingredients.append(linea)
#         final_ingredients = [linea for linea in lineas_limpias_final]

#         elem['Ingredientes_pr'] = limpiar_especificaciones(final_ingredients, palabras_eliminar)
#         elem['Ingredientes_pr'] = [singularize_ingredient(ing).rstrip('.') for ing in elem['Ingredientes_pr']]
#         elem['Ingredientes_pr'] = [limpiar_linea(ing, pattern).lower().rstrip('.') for ing in elem['Ingredientes_pr']]
#         elem['Ingredientes_pr'] = remove_empty_strings(elem['Ingredientes_pr'])

#         elem['Ingredientes_pr'] = limpiar_especificaciones(
#             elem['Ingredientes_pr'],
#             ['cc de ', '. de ', '. ', '- ', 'c/n ', ' c/n', 'ajã\xade', 'ajã\xad ', 'ðÿ§…', ' %',
#              'taza ', 'centimetro cubico de ', 'c.c', 'una ', 'trozo de ', 'un kilo de ',
#              ' en cuadradito', 'semilla de ', 'copita de ']
#         )

#         elem['Ingredientes_pr'] = [unidecode(ing) for ing in elem['Ingredientes_pr']]

#     return recipes_dict



# def get_ingredient_text(recipes_dict,ai_generated=False):

#     # This function extracts the ingredient text from the recipes
#     """ 
#     Args:
#     recipes_dict: list of dictionaries with the recipes
#     ai_generated: boolean to indicate if the recipes are AI generated or not
#     Returns:
#     recipes_dict: list of dictionaries with the recipes and the ingredient text
#     """ 

#     for i,elem in enumerate(recipes_dict):
#         if ai_generated:
#             final_text = (elem['mod'].split("Ingredientes")[1]).split("\n\n")[1].replace(" ", " ")
            
#         else:
#             if pd.isna(elem['Ingredientes']):
#                 final_text = ''

#             elif elem['Ingredientes'].strip().startswith("["): #some cases they save the ingredients as a list
#                 # Safely evaluate the string to convert it to a list
#                 ingredientes = ast.literal_eval(elem['Ingredientes'])
#                 # lowercase
#                 # join elements of the list in a text
#                 final_text = ', '.join(ingredientes)  
#             else:
#                 final_text = elem['Ingredientes']
                
#         final_text = final_text.lower()
#         elem['ingredient_text'] = unidecode(final_text)
#     return recipes_dict
def clean_ingredients(recipes_dict, ai_generated=False, list_mode=False, name_column='mod'):
    remove_after_word = [' entre', ' al gusto', ' para', 'para ']
    palabras_eliminar = [
        "mediana", "bastantes", "bastante", "rebanadas de", "pequeña", "trozos medianos",
        "al gusto", "a gusto", "trozos de ", "trozo de ", "en trozos pequeños", ' ya ',
        " ()", "pequeño", "regulares", "lb. ", " suficiente"
    ]

    pattern = r'\b\d+\s*(?:\/\d+)?\s*(?:g|gramos|pieza|centímetro cúbico|cucharada sopera|cc|un kilo|tableta|pisca|piezas|paquete|paquetes|tarro|loncha|cabeza|cabezas|lonchas|kilogramo|kilogramos|puñados|chorrito|botella|gr.|libra|ralladura|lámina|barra|caja|rama|puñado|manojo|bote|vaso|pellizco|unidad|chorro|lata|postre|litro|litros|mililitros|barra|-|cucharada colmada|unidades|copa|kilo|gr|kg|ml|cl|dl|l|cm|bolas|dientes|diente|pizca|cucharadas soperas|tiras|tajadas|cucharaditas|cucharadita|cucharadas|mix|cucharada|cc|cda|cdta)?(?:\s+(?!de\b)(?:taza|tazas))?\b\s*(?:de)?'

    for i, elem in enumerate(recipes_dict):
        # print(f"Procesando receta {i}")

        if ai_generated:
            if list_mode:
                ingredientes_totales = []

                for mod_text in elem.get("mod_list", []):
                    texto = mod_text.split("Ingredientes", 1)[-1]
                    texto = texto.split("Pasos", 1)[0].strip()
                    texto = texto.lstrip(": ").strip().lower()

                    texto = texto.replace("¼", "1/4").replace("½", "1/2").replace("¾", "3/4").replace("⅓", "1/3")
                    texto = texto.replace("al gusto,", "").replace("al gusto:", "")

                    if "* " in texto:
                        ingredientes = texto.split("* ")
                    elif "\n" in texto:
                        ingredientes = texto.split("\n")
                    elif "." in texto and texto.count('.') > texto.count(','):
                        ingredientes = texto.split(".")
                    else:
                        ingredientes = [i.strip() for i in re.split(r"\n|,|\.", texto) if i.strip()]

                    ingredientes_principal = []
                    for ing in ingredientes:
                        ing = ing.strip()
                        if not ing:
                            continue
                        if " sin " in ing and "," not in ing:
                            ing = ing.split(" sin ")[0].strip()
                        ing = re.sub(r"\([^)]*\)", "", ing).strip()
                        ingredientes_principal.append(ing)

                    lineas_limpias = [limpiar_linea(linea, pattern).lower() for linea in ingredientes_principal if linea.strip()]
                    lineas_limpias_final = [re.sub(r'^(taza de|tazas de)\s+', '', linea, flags=re.IGNORECASE) for linea in lineas_limpias]

                    final_ingredients = [linea for linea in lineas_limpias_final]
                    ingredientes_limpios = limpiar_especificaciones(final_ingredients, palabras_eliminar)
                    ingredientes_limpios = [singularize_ingredient(ing).rstrip('.') for ing in ingredientes_limpios]
                    ingredientes_limpios = [limpiar_linea(ing, pattern).lower().rstrip('.') for ing in ingredientes_limpios]
                    ingredientes_limpios = remove_empty_strings(ingredientes_limpios)

                    ingredientes_limpios = limpiar_especificaciones(
                        ingredientes_limpios,
                        ['cc de ', '. de ', '. ', '- ', 'c/n ', ' c/n', 'ajã\xade', 'ajã\xad ', 'ðÿ§…', ' %',
                         'taza ', 'centimetro cubico de ', 'c.c', 'una ', 'trozo de ', 'un kilo de ',
                         ' en cuadradito', 'semilla de ', 'copita de ']
                    )

                    ingredientes_limpios = [unidecode(ing) for ing in ingredientes_limpios]
                    ingredientes_totales.append(ingredientes_limpios)

                elem['Ingredientes_pr'] = ingredientes_totales

            else:
                texto = elem.get(name_column, "")
                texto = texto.split("Ingredientes", 1)[-1]
                texto = texto.split("Pasos", 1)[0].strip()
                texto = texto.lstrip(": ").strip().lower()

                texto = texto.replace("¼", "1/4").replace("½", "1/2").replace("¾", "3/4").replace("⅓", "1/3")
                texto = texto.replace("al gusto,", "").replace("al gusto:", "")

                if "* " in texto:
                    ingredientes = texto.split("* ")
                elif "\n" in texto:
                    ingredientes = texto.split("\n")
                elif "." in texto and texto.count('.') > texto.count(','):
                    ingredientes = texto.split(".")
                else:
                    ingredientes = [i.strip() for i in re.split(r"\n|,|\.", texto) if i.strip()]

                ingredientes_principal = []
                for ing in ingredientes:
                    ing = ing.strip()
                    if not ing:
                        continue
                    if " sin " in ing and "," not in ing:
                        ing = ing.split(" sin ")[0].strip()
                    ing = re.sub(r"\([^)]*\)", "", ing).strip()
                    ingredientes_principal.append(ing)

                lineas_limpias = [limpiar_linea(linea, pattern).lower() for linea in ingredientes_principal if linea.strip()]
                lineas_limpias_final = [re.sub(r'^(taza de|tazas de)\s+', '', linea, flags=re.IGNORECASE) for linea in lineas_limpias]

                final_ingredients = [linea for linea in lineas_limpias_final]
                ingredientes_limpios = limpiar_especificaciones(final_ingredients, palabras_eliminar)
                ingredientes_limpios = [singularize_ingredient(ing).rstrip('.') for ing in ingredientes_limpios]
                ingredientes_limpios = [limpiar_linea(ing, pattern).lower().rstrip('.') for ing in ingredientes_limpios]
                ingredientes_limpios = remove_empty_strings(ingredientes_limpios)

                ingredientes_limpios = limpiar_especificaciones(
                    ingredientes_limpios,
                    ['cc de ', '. de ', '. ', '- ', 'c/n ', ' c/n', 'ajã\xade', 'ajã\xad ', 'ðÿ§…', ' %',
                     'taza ', 'centimetro cubico de ', 'c.c', 'una ', 'trozo de ', 'un kilo de ',
                     ' en cuadradito', 'semilla de ', 'copita de ']
                )

                elem['Ingredientes_pr'] = [unidecode(ing) for ing in ingredientes_limpios]

        else:
            if pd.isna(elem['Ingredientes']):
                elem['Ingredientes_pr'] = ''
                continue

            if elem['Ingredientes'].strip().startswith("["):
                ingredientes = ast.literal_eval(elem['Ingredientes'])
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
                texto = texto.lower()
                texto = texto.replace("¼", "1/4").replace("½", "1/2").replace("¾", "3/4").replace("⅓", "1/3")
                texto = texto.replace("al gusto,", "").replace("al gusto:", "")
                texto = texto.replace("cta.de", "").replace("cinco", "5").replace("azãºcar", "azúcar")
                ingredientes_principal = texto.split(',')

            lineas_limpias = [limpiar_linea(linea, pattern).lower() for linea in ingredientes_principal if linea.strip()]
            lineas_limpias_final = [re.sub(r'^(taza de|tazas de)\s+', '', linea, flags=re.IGNORECASE) for linea in lineas_limpias]

            final_ingredients = [linea for linea in lineas_limpias_final]
            ingredientes_limpios = limpiar_especificaciones(final_ingredients, palabras_eliminar)
            ingredientes_limpios = [singularize_ingredient(ing).rstrip('.') for ing in ingredientes_limpios]
            ingredientes_limpios = [limpiar_linea(ing, pattern).lower().rstrip('.') for ing in ingredientes_limpios]
            ingredientes_limpios = remove_empty_strings(ingredientes_limpios)

            ingredientes_limpios = limpiar_especificaciones(
                ingredientes_limpios,
                ['cc de ', '. de ', '. ', '- ', 'c/n ', ' c/n', 'ajã\xade', 'ajã\xad ', 'ðÿ§…', ' %',
                 'taza ', 'centimetro cubico de ', 'c.c', 'una ', 'trozo de ', 'un kilo de ',
                 ' en cuadradito', 'semilla de ', 'copita de ']
            )

            elem['Ingredientes_pr'] = [unidecode(ing) for ing in ingredientes_limpios]

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
def clean_single_ingredient(ingredientes, ai_generated=False):
    remove_after_word = [' entre', ' al gusto', ' para', 'para ']
    palabras_eliminar = ["mediana", "bastantes", "bastante", "rebanadas de", "pequeña", "trozos medianos", 
                         "al gusto", "a gusto", "trozos de ", "trozo de ", "en trozos pequeños", ' ya ',
                         " ()", "pequeño", "regulares", "lb. ", " suficiente"]

    pattern = r'\b\d+\s*(?:\/\d+)?\s*(?:g|gramos|pieza|centímetro cúbico|cucharada sopera|cc|un kilo|tableta|pisca|piezas|paquete|paquetes|un kilo|tarro|loncha|cabeza|cabezas|lonchas|kilogramo|kilogramos|puñados|chorrito|botella|gr.|libra|ralladura|lámina|barra|paquete|bastante|caja|rama|puñado|manojo|bote|vaso|pellizco|unidad|chorro|vaso|lata|rama|postre|litro|litros|mililitros|barra|-|cucharada colmada|unidades|copa|un kilo|kilo|gr|kg|ml|cl|dl|l|cm|bolas|dientes|diente|pizca|cucharadas soperas|tiras|tajadas|cucharaditas|cucharadita|cucharadas|mix|cucharada|cc|cda|cdta)?(?:\s+(?!de\b)(?:taza|tazas))?\b\s*(?:de)?'

    if ai_generated:
        texto = ingredientes.split("Ingredientes")[1].split("\n\n")[1].replace(" ", " ")
        texto = texto.lower()
        texto = texto.replace("¼", "1/4").replace("½", "1/2").replace("¾", "3/4").replace("⅓", "1/3")
        texto = texto.replace("Al gusto,", "").replace("Al gusto:", "")
        ingredientes_principal = [i.split(',')[0] for i in texto.split("* ")]
        ingredientes_principal = [i.split('(')[0] for i in ingredientes_principal]
    else:
        if ingredientes.strip().startswith("["):
            ingredientes = ast.literal_eval(ingredientes)
            ingredientes = [ing.lower() for ing in ingredientes]
            ingredientes_principal = [remove_text_in_parentheses(ing) for ing in ingredientes]
            ingredientes_principal = [remove_text_after_coma(ing) for ing in ingredientes_principal]
            ingredientes_principal = [remove_everything_after(remove_after_word, ing) for ing in ingredientes_principal]
        else:
            texto = remove_text_in_parentheses(ingredientes)
            texto = remove_text_after_coma(texto)
            texto = remove_everything_after(remove_after_word, texto)
            texto = texto.lower()
            texto = texto.replace("¼", "1/4").replace("½", "1/2").replace("¾", "3/4").replace("⅓", "1/3")
            texto = texto.replace("al gusto,", "").replace("al gusto:", "")
            texto = texto.replace("cta.de", "").replace("cinco", "5").replace("azãºcar", "azúcar")
            ingredientes_principal = texto.split(',')
            

    lineas_limpias = [limpiar_linea(linea, pattern).lower() for linea in ingredientes_principal if linea.strip()]
    lineas_limpias_final = [re.sub(r'^(taza de|tazas de)\s+', '', linea, flags=re.IGNORECASE) for linea in lineas_limpias]

    final_ingredients = []
    for i in lineas_limpias_final:
        if ' y ' in i:
            final_ingredients.append(i.split(' y ')[0])
        elif ' o ' in i:
            final_ingredients.append(i.split(' o ')[0])
        else:
            final_ingredients.append(i)

    ingredientes_limpios = limpiar_especificaciones(final_ingredients, palabras_eliminar)
    ingredientes_limpios = [singularize_ingredient(ingredient).rstrip('.') for ingredient in ingredientes_limpios]
    ingredientes_limpios = [limpiar_linea(linea, pattern).lower().rstrip('.') for linea in ingredientes_limpios]
    ingredientes_limpios = remove_empty_strings(ingredientes_limpios)

    ingredientes_limpios = limpiar_especificaciones(
        ingredientes_limpios,
        ['cc de ','. de ', '. ', '- ', 'c/n ', ' c/n', 'ajã\xade', 'ajã\xad ', 'ðÿ§…', ' %', 'taza ',
         'centimetro cubico de ', 'c.c', 'una ', 'trozo de ', 'un kilo de ', ' en cuadradito', 'semilla de ',
         'copita de ']
    )

    ingredientes_limpios = [unidecode(ing) for ing in ingredientes_limpios]

    return ingredientes_limpios


