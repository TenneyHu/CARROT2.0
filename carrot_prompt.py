from llama_index.core import PromptTemplate

recipe_cultural_adaption_select_template_str = (
    "Your task is to find the Spanish recipe that is most relevant to the given recipe.\n"
    "Below are some Spanish recipes retrieved through the search:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the source recipe title: {query_str}, select the most relevant answer from the context.\n"
    "NOTE: If there are multiple highly relevant recipes, output the one that best aligns with the cultural preferences and norms of Spanish users.\n"
    "NOTE: If no relevant recipe is found in the context, generate a recipe that aligns with Spanish dietary habits based on the given recipe. \n"
    "NOTE: the recipe you output should include the full recipe title, ingredients, and steps, following the same format as the context. \n"
    "NOTE: Do not output any content other than the recipe. \n"
    "Best Answer:"
)

recipe_cultural_adaption_refine_template_str = (
    "Your task is to find the Spanish recipe that is most relevant to the given recipe.\n"
    "Below are some Spanish recipes retrieved through the search:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the source recipe title: {query_str}, select the most relevant answer from the context.\n"
    "NOTE: If there are multiple highly relevant recipes, output the one that best aligns with the cultural preferences and norms of Spanish users.\n"
    "NOTE: If no relevant recipe is found in the context, generate a recipe that aligns with Spanish dietary habits based on the given recipe. \n"
    "NOTE: the recipe you output should include the full recipe title, ingredients, and steps, following the same format as the context. \n"
    "NOTE: Do not output any content other than the recipe. \n"
    "Best Answer:"
)

def load_prompt(task):
    prompts_dict = {}
    
    if task == "recipe adaption":
        prompts_dict["text_qa_template"] = PromptTemplate(recipe_cultural_adaption_select_template_str)
        prompts_dict["refine_template"] = PromptTemplate(recipe_cultural_adaption_refine_template_str)
    
    return prompts_dict

