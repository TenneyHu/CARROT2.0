
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from transformers import AutoModel
from tqdm import tqdm
import re
from unidecode import unidecode
import inflect
import ast

p = inflect.engine()

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

def clean_ingredients(recipes, ai_generated=False):

    remove_after_word = [' entre', ' al gusto', ' para', 'para ']
    palabras_eliminar = ["mediana", "bastantes", "bastante", "rebanadas de", "pequeña", "trozos medianos", 
                     "al gusto", "a gusto", "trozos de ", "trozo de ", "en trozos pequeños", ' ya ',
                       " ()", "pequeño", "regulares", "lb. ", " suficiente"]
    ingredient_list = []
    for elem in recipes:

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
            if elem.strip().startswith("["):
                try:
                    ingredientes = ast.literal_eval(elem.strip())
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
                except:
                    continue
            else:
                texto = remove_text_in_parentheses(elem)
                texto = remove_text_after_coma(texto)
                texto = remove_everything_after(remove_after_word, texto)
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

        res = limpiar_especificaciones(final_ingredients, palabras_eliminar)
        res = [singularize_ingredient(ingredient).rstrip('.') for ingredient in res ]
        res  = [limpiar_linea(linea,pattern).lower().rstrip('.') for linea in res]
        res = remove_empty_strings(res)
        res = limpiar_especificaciones(res, ['cc de ','. de ', '. ', '- ', 'c/n ', ' c/n', 'ajã\xade', 'ajã\xad ', 'ðÿ§…', ' %', 'taza ', 'centimetro cubico de ', 'c.c', 'una ', 'trozo de ', 'un kilo de ', ' en cuadradito', 'semilla de ', 'copita de '])
        # unidecode
        res= [unidecode(ing) for ing in res]
        ingredient_list.extend(res)
    return ingredient_list


def calc_avg_semantic_diversity(model, qid_to_texts):
    diversities = []
    for texts in qid_to_texts.values():
        if len(texts) < 2:
            continue  
        embeddings = model.encode(texts, show_progress_bar=False)
        similarity_matrix = cosine_similarity(embeddings)
        upper_triangular = similarity_matrix[np.triu_indices(len(texts), k=1)]
        diversity = 1 - np.mean(upper_triangular)
        diversities.append(diversity)
    
    return np.mean(diversities) if diversities else 0

def calc_global_diversity(ingredients):
    ingredient_list =  clean_ingredients(ingredients, ai_generated=False)
    global_diversity = len(set(ingredient_list))
    return global_diversity
