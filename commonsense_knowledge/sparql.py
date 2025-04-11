
import time
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import random
import re
from tqdm import tqdm
# Set up the DBpedia SPARQL endpoint URL
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# download data from Wikidata
def safe_query(sparql):
    """Retries SPARQL queries up to 5 times in case of a server error."""
    for attempt in range(5):  # Retry up to 5 times
        try:
            return sparql.query().convert()
        except:
            print(f"Retrying in {2**attempt} seconds...")
            time.sleep(2**attempt)  # Exponential backoff delay
    print("Query failed after 5 attempts.")
    return None


def get_dish(countries):
    # Function to fetch dishes from Wikidata
    # This function fetches dishes from Wikidata based on the provided country IDs.
    # It constructs a SPARQL query to retrieve dish information, including name, description, ingredients, and origin.
    # The results are stored in a list of dictionaries, which is then converted to a DataFrame and saved as a CSV file.
    DISHES = []
    Dish_Count = {}

    for country, countryID in tqdm(countries.items()):
        print(f"Fetching dishes for: {country}")
        query = f"""
        SELECT DISTINCT ?dish ?dishLabel ?description ?origin
        (GROUP_CONCAT(DISTINCT ?ingredientLabel; separator=", ") AS ?hasParts)
        (GROUP_CONCAT(DISTINCT ?MadeFromMateriaLabel; separator=", ") AS ?MadeFromMateria)
        (GROUP_CONCAT(DISTINCT ?image; separator=", ") AS ?imageLabel)
        WHERE {{
          ?dish (wdt:P31|wdt:P279)+ wd:Q746549.
          {{?dish wdt:P495 wd:{countryID}.}}
          UNION
          {{?dish (wdt:P2012|wdt:P361) [(wdt:P17|wdt:P495) wd:{countryID}].}}

          ?dish rdfs:label ?dishLabel .
          FILTER(LANG(?dishLabel) = "es")  # ✅ Keep only Spanish dish names

          OPTIONAL {{ ?dish schema:description ?description . FILTER(LANG(?description) = "es") }}  # ✅ Keep only Spanish descriptions
          OPTIONAL {{ ?dish wdt:P527 ?ingredient . ?ingredient rdfs:label ?ingredientLabel . FILTER(LANG(?ingredientLabel) = "es") }}  # ✅ Keep only Spanish ingredients
          OPTIONAL {{ ?dish wdt:P18 ?image. ?dish wdt:P186 ?Materia. ?Materia rdfs:label ?MadeFromMateriaLabel . FILTER(LANG(?MadeFromMateriaLabel) = "es") }}  # ✅ Keep only Spanish material

          wd:{countryID} rdfs:label ?origin
          FILTER(LANG(?origin) = "es")  # ✅ Keep only Spanish country names
        }}
        GROUP BY ?dish ?dishLabel ?description ?origin
        """  # Limiting to 50 results

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        time.sleep(2)  # Avoid query overload

        results = safe_query(sparql)
        if results is None:
            continue  # Skip this country if query fails

        Dish_Count[country] = len(results["results"]["bindings"])
        print(f"{Dish_Count[country]} dishes found for {country}")

        for result in results["results"]["bindings"]:
            dish = {
                "url": result["dish"]["value"],
                "origin": country,
                "origin_name": result.get("origin", {}).get("value", ""),
                "name": result["dishLabel"]["value"],
                "description": result.get("description", {}).get("value", ""),
                "ingredients": result.get("hasParts", {}).get("value", ""),
                "made_from": result.get("MadeFromMateria", {}).get("value", ""),
                "image": result.get("imageLabel", {}).get("value", ""),
            }
            DISHES.append(dish)
        # save partial results
        # df = pd.DataFrame(DISHES)
        # df.to_csv('dishes_partial.csv', index=False)
    # Save results
    df = pd.DataFrame(DISHES)
    df.to_csv('dishes.csv', index=False)  # ✅ Save filtered data

    return Dish_Count, df


# Function to clean scientific names and non-ingredient references
def clean_ingredients(text):
    if pd.isna(text) or not isinstance(text, str):
        return text  # Ignore NaN or non-string values

    # Normalize separators in case some items use semicolons or other symbols
    text = text.replace(";", ",")  # Convert semicolons to commas if needed
    items = [item.strip() for item in text.split(",")]

    # Enhanced regex for scientific names (handles `var.`, `subsp.`, `f.` cases)
    scientific_pattern = re.compile(r"^[A-Z][a-z]+(?:\s[a-z]+)+(?:\s(var\.|subsp\.|f\.)\s[a-z]+)*$")

    # Latin suffixes common in scientific names
    latin_suffixes = ["aceae", "ales", "inae", "idae", "iformes", "phora", "phyllum",
                      "sicum", "cicla", "ensis", "ifolia", "anae", "phyta", "mycota",
                      "bacteria", "viridae", "phyceae", "mycetes", "ium"]

    # Explicit list of non-ingredient references
    non_ingredient_patterns = [
        r"^Anexo:.*",  # Matches anything starting with "Anexo:"
        r"\bvalor culinario\b",  # Matches "valor culinario" anywhere
        r"\b(lista|clasificación|categoría|índice|documento|sección|referencia)\b",  # Non-food words
        r"http[s]?://",  # Matches URLs
    ]

    cleaned_items = []
    for item in items:
        item = item.strip()  # Remove extra spaces
        
        # Remove scientific names
        if scientific_pattern.match(item) or any(item.endswith(suffix) for suffix in latin_suffixes):
            continue  # Skip scientific names

        # Remove explicit non-ingredient references
        if any(re.search(pattern, item, re.IGNORECASE) for pattern in non_ingredient_patterns):
            continue  # Skip references

        cleaned_items.append(item)

    # Return cleaned list as a string, or NaN if empty
    return ", ".join(cleaned_items) if cleaned_items else None 





# Function to clean scientific names and non-ingredient references
def clean_ingredients(text):
    
    if pd.isna(text) or not isinstance(text, str):
        return text  # Ignore NaN or non-string values

    # Normalize separators in case some items use semicolons or other symbols
    text = text.replace(";", ",")  # Convert semicolons to commas if needed
    items = [item.strip() for item in text.split(",")]

    # Enhanced regex for scientific names (handles `var.`, `subsp.`, `f.` cases)
    scientific_pattern = re.compile(r"^[A-Z][a-z]+(?:\s[a-z]+)+(?:\s(var\.|subsp\.|f\.)\s[a-z]+)*$")

    # Latin suffixes common in scientific names
    latin_suffixes = ["aceae", "ales", "inae", "idae", "iformes", "phora", "phyllum",
                      "sicum", "cicla", "ensis", "ifolia", "anae", "phyta", "mycota",
                      "bacteria", "viridae", "phyceae", "mycetes", "ium"]

    # Explicit list of non-ingredient references
    non_ingredient_patterns = [
        r"^Anexo:.*",  # Matches anything starting with "Anexo:"
        r"\bvalor culinario\b",  # Matches "valor culinario" anywhere
        r"\b(lista|clasificación|categoría|índice|documento|sección|referencia)\b",  # Non-food words
        r"http[s]?://",  # Matches URLs
    ]

    cleaned_items = []
    for item in items:
        item = item.strip()  # Remove extra spaces
        
        # Remove scientific names
        if scientific_pattern.match(item) or any(item.endswith(suffix) for suffix in latin_suffixes):
            continue  # Skip scientific names

        # Remove explicit non-ingredient references
        if any(re.search(pattern, item, re.IGNORECASE) for pattern in non_ingredient_patterns):
            continue  # Skip references

        cleaned_items.append(item)

    # Return cleaned list as a string, or NaN if empty
    return ", ".join(cleaned_items) if cleaned_items else None 




def remove_latin_words(text):
    """Removes scientific names and Latin-derived words while preserving proper formatting, including compound words."""
    if not isinstance(text, str):  # Skip NaN values
        return text

    # Normalize spacing before splitting
    words = [word.strip() for word in text.split(",")]

    cleaned_words = [
        word for word in words
        if not re.fullmatch(r'^[A-Z][a-z]+(?: [a-z]+)*$', word)  # Scientific names (single or multi-word)
        and not re.search(r'\b[a-zA-Z-]+(?:um|us|ae)\b', word)  # Latin-derived words ending in -um, -us, -ae
    ]
    
    return ", ".join(cleaned_words) if cleaned_words else None  # Return cleaned text





def generate_options(correct, all_options, num_distractors=3):
    # Define function to generate multiple-choice options
    # Ensure the correct answer is included
    # and distractors are unique and not the correct answer

    distractors = random.sample([opt for opt in all_options if opt != correct], min(num_distractors, len(all_options)-1))
    options = [correct] + distractors
    random.shuffle(options)  # Shuffle to randomize order
    return options

def create_questions(df):
    # Function to create questions based on dish data
    # This function generates questions about the main ingredient and the country of origin for each dish.
    # It uses the cleaned ingredient data and generates multiple-choice options for each question.
    # Initialize an empty list to store questions

    questions = []

    # Collect all possible unique ingredients correctly
    all_possible_ingredients = set()
    for ing_list in df["ingredients"].dropna():
        ing_list = ing_list.split(", ")  # Ensure correct splitting
        all_possible_ingredients.update(ing_list)  # Add unique ingredients

    all_possible_ingredients = list(all_possible_ingredients)  # Convert back to list

    # Generate questions from dishes
    for _, row in df_dishes.iterrows():
        dish = row["name"]
        origin = row["origin"]
        # Ensure ingredients are properly formatted
        ingredients = remove_latin_words(row["ingredients"]) if isinstance(row["ingredients"], str) else ""

        # Apply the same cleaning to "made_from"
        made_from = remove_latin_words(row["made_from"]) if isinstance(row["made_from"], str) else ""

        # Convert cleaned strings back to lists
        ingredients = ingredients.split(", ") if ingredients else []
        made_from = made_from.split(", ") if made_from else []

        # Combine both lists
        all_ingredients = ingredients + made_from

        # Generate ingredient-based question
        if all_ingredients:
            correct_answer = all_ingredients[0]

            options = generate_options(correct_answer, all_possible_ingredients)
            
            questions.append({
                "Question": f"¿Cuál es un ingrediente principal de {dish} en {origin}?",
                "Answer Options": options,
                "Correct Answer": correct_answer,
                "Dish": dish,
                "Origin": origin,
                "Question type": "main ingredient"
            })
        
        # Generate cultural question
        questions.append({
            "Question": f"¿En qué país es tradicionalmente popular el plato {dish}?",
            "Answer Options": generate_options(origin, list(set(df_dishes["origin"]))),
            "Correct Answer": origin,
            "Dish": dish,
            "Origin": origin,
            "Question type": "country of traditional dish"
        })

    
    # Convert questions list to DataFrame
    df_questions = pd.DataFrame(questions)

    # Save the dataset as CSV
    df_questions.to_csv("wikidata_questions.csv", index=False)

    return df_questions




# create main function
if __name__ == "__main__":

    # Download from wikidata
    countries = {'España': 'Q29','México':'Q96', 'Argentina':'Q414','Colombia':'Q739','Venezuela':'Q717','Perú':'Q419'}
    dish_counts, df_dishes = get_dish(countries)
    df_dishes.to_csv('dishes.csv', index=False) 
    print(df_dishes.shape)
    print(df_dishes['origin'].value_counts())
    df_dishes.head(4)


    random.seed(42)
    # Load your dish dataset (ensure it's correctly formatted)
    # df_dishes = pd.read_csv("dishes.csv")

    df = df_dishes.copy()
    # Apply cleaning function to 'ingredients' and 'made_from' columns
    df["ingredients"] = df["ingredients"].apply(clean_ingredients)
    df["made_from"] = df["made_from"].apply(clean_ingredients)

    # get the list of ingredient_cleaned
    ingredients_list = df['ingredients'].dropna().str.split(', ').sum()
    print(ingredients_list)

    # Example Usage
    text = "Rubus fruticosus, album-verum, Cucumis, Basilicum, patata, Tote asilum"
    print(remove_latin_words(text))  # Expected output: "Cucumis, patata"

    # Create questions
    df_questions = create_questions(df)
    df_questions[df_questions['Question type']=='main ingredient'].head(5)
