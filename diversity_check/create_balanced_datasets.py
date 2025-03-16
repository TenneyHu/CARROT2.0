import pandas as pd
# from collections import defaultdict
from sklearn.utils import shuffle
import numpy as np
np.random.seed(33)
from huggingface_hub import hf_hub_download
from metrics import clean_ingredients
import json
from diversity_metrics import compute_lexical_diversity

# ______________________________________________________________________________________________________________________
def create_balanced_dataset(dataset):
    # from all the countries, we select 200 items per country to have a balanced dataset
    # we also shuffle the dataset

    # get all the countries in the dataset
    countries = dataset['Pais'].unique()
    print("Countries in the dataset:", countries)

    # create a list to store the balanced dataset
    balanced_dataset = pd.DataFrame()
    for country in countries:
        country_data = dataset[dataset['Pais'] == country]

        if country_data.shape[0] > 200:
            country_data = shuffle(country_data)
            country_data = country_data.head(200)
            balanced_dataset = pd.concat([balanced_dataset, country_data])

    # Count the values in the 'Category' column
    print(balanced_dataset['Pais'].value_counts())

    # get list of id per country for getting the ai-generated versions
    country_ids = []
    for country in countries:
        country_data = dataset[dataset['Pais'] == country]
        country_ids.extend(balanced_dataset['Id'].values.tolist())

    balanced_dataset.to_csv("res/balanced_recipes.csv", index=False)

    return balanced_dataset
# ______________________________________________________________________________________________________________________


def generate_balanced_adapted_dataset(list_tmp = ['1e-13','0.3','0.6','0.9']):

    # create empty df
    df = pd.DataFrame()

    for tmp in list_tmp:
        print("Computing temperature value",tmp)
        with open('res/output_'+tmp+'.json', 'r') as file:
            data = json.load(file)
            #  concat json to the df 
            df_data = pd.DataFrame(data)
            df_data['temperature'] = tmp
            df = pd.concat([df, df_data])

    df_expanded = df['src'].apply(pd.Series)

    # Join the expanded DataFrame back to the original DataFrame with suffixes
    df = df.join(df_expanded, rsuffix='_expanded')

    #remove columns by name
    df = df.drop(columns=['country_expanded', 'id_expanded', 'src'])
    # change name of column
    df.rename(columns={'src_expanded': 'src'}, inplace=True)
    df.head()

    # unify peru and perú
    df["country"] = df["country"].replace("Perú", "Peru")


    list_tmp = ['1e-13', '0.3', '0.6', '0.9']
    balanced_dataset = pd.DataFrame()

    # Step 1: Process the first temperature to collect 200 recipes per country
    list_ids = {}  # Initialize as an empty dictionary

    for tmp in list_tmp[:1]:
        print(f"Processing temperature: {tmp}")
        
        # Get DataFrame with the specified temperature
        df_tmp = df[df['temperature'] == tmp]
        # Get list of countries
        countries = df_tmp['country'].unique()
        
        for country in countries:
            country_data = df_tmp[df_tmp['country'] == country]

            #remove duplicates
            country_data = country_data.drop_duplicates(subset=['id'])
            
            if country_data.shape[0] > 200:
                print(f"{country}: {country_data.shape[0]} recipes")
                # Shuffle and select 200 random recipes
                country_data = shuffle(country_data)
                country_data = country_data.head(200)
                
                # Store the recipe IDs for the country in the dictionary
                list_ids[country] = country_data['id'].values.tolist()
                balanced_dataset = pd.concat([balanced_dataset, country_data], ignore_index=True)


    # Step 2: Process the remaining temperatures using the same recipe IDs
    for tmp in list_tmp[1:]:
        
        # Get DataFrame with the specified temperature
        df_tmp = df[df['temperature'] == tmp]
        
        for country, ids in list_ids.items():
            print(f"Processing country: {country}")
            # Filter recipes for the country with the specific IDs
            country_data = df_tmp[(df_tmp['country'] == country)]
                                
            #filter by ID
            country_data = country_data[country_data['id'].isin(ids)]
            #remove duplicates
            country_data = country_data.drop_duplicates(subset=['id'])  

            balanced_dataset = pd.concat([balanced_dataset, country_data], ignore_index=True)


    # Group by Country and Temperature, then count the number of recipes
    final_counts = balanced_dataset.groupby(['country', 'temperature']).size().reset_index(name='Count')

    # Display the final counts
    print(final_counts)
    
    # Save the balanced dataset to a CSV file
    balanced_dataset.to_csv("res/balanced_adapted_recipes.csv", index=False)

    print("Balanced dataset created successfully!")
    print("Processing original recipes...")
    original_recipes = pd.DataFrame()
    for country, ids in list_ids.items():
        print(".............................")
        print(f"Processing country: {country}")
        print(f"Number of recipes: {len(ids)}")
        # Filter recipes for the country with the specific IDs
        country_data = dataset[(dataset['Pais'] == country)]
                                
        #filter by ID
        country_data = country_data[country_data['Id'].isin(ids)]
        #remove duplicates
        country_data = country_data.drop_duplicates(subset=['Id'])  

        #concat to original recipes
        original_recipes = pd.concat([original_recipes, country_data], ignore_index=True)


    # we also need to include Spanish datasets. This is not included in the previous dataset because the adapted recipes doesnt consider spanish. 
    country_data = dataset[(dataset['Pais'] == 'Spain')]
    #shuffle and get 200
    country_data = shuffle(country_data)
    country_data_extended = country_data.head(1200) # this is for comparing large scale spanish/non spanish only. 
    country_data = country_data.head(200)
    original_recipes = pd.concat([original_recipes, country_data], ignore_index=True)

    # Group by Country and Temperature, then count the number of recipes
    final_counts = original_recipes.groupby('Pais').size().reset_index(name='Count')
    # Display the final counts
    print(final_counts)

    # change name to columns
    original_recipes.rename(columns={'Id': 'id', 'Pais': 'country'}, inplace=True)
    original_recipes['temperature'] = 'original'
    original_recipes.head(1)

    # save original recipes
    original_recipes.to_csv("res/balanced_original_recipes.csv", index=False)

    return balanced_dataset, original_recipes
# ______________________________________________________________________________________________________________________


# create main function
REPO_ID = "YOUR_REPO_ID"
FILENAME = "data.csv"

dataset = pd.read_csv(
    hf_hub_download(repo_id='somosnlp/RecetasDeLaAbuela', filename='recetasdelaabuela.csv', 
                    repo_type="dataset")
)

print("Running unification of recipes from different countries...")
dataset["Pais"] = dataset["Pais"].replace("Perú", "Peru")
dataset["Pais"] = dataset["Pais"].replace("España", "Spain")
dataset["Pais"] = dataset["Pais"].replace("Internacional", "International")


# balanced_original_dataset = create_balanced_dataset(dataset)
balanced_adapted_dataset, balanced_original_dataset = generate_balanced_adapted_dataset()
# print(balanced_adapted_dataset.shape, "recipes from Spain")
# print("Visualize the first two recipes of the balanced dataset:")
# print(balanced_adapted_dataset.head(2))
# print("\n")
# print(balanced_original_dataset.shape, "recipes from Spain")
# print("Visualize the first two recipes of the balanced dataset:")
# print(balanced_original_dataset.head(2))






# # # results_analysis = compute_lexical_diversity(original_recipes,separated_ingredients=False) #one token one word
# # # print("One token one word", results_analysis) # here we consider each word as a token in the list. notice that one ingredient can have multiple words, and we are separating this. This makes the diversity pretty low. 

# # results_analysis = compute_lexical_diversity(original_recipes,separated_ingredients=True) #one token one ingredient. Here we consider each ingredient as a token in the list. This makes the diversity higher.
# # results_analysis

# compute_lexical_diversity(original_recipes,separated_ingredients=True,mode="ingredient_text") 
# compute_lexical_diversity(original_recipes,separated_ingredients=True,mode="ingredient_token")
compute_lexical_diversity(balanced_original_dataset,separated_ingredients=True, mode="whole_recipe") #one token one ingredient. Here we consider each ingredient as a token in the list. This makes the diversity higher.
