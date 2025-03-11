from llama_index.core import PromptTemplate

recipe_cultural_adaption_select_template_str = (
    "Your task is to find the Spanish recipe that is most relevant to the given recipe.\n"
    "Below are some Spanish recipes retrieved through the search:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the source recipe title: {query_str}, select the most relevant recipe from the context.\n"
    "\n"
    "Instructions:\n"
    "- If there are multiple highly relevant recipes, select the one that best aligns with the cultural preferences and norms of Spanish cuisine.\n"
    "- The output must contain a complete recipe, including:\n"
    "  - Nombre: The name of the dish.\n"
    "  - Ingredientes: A detailed list of ingredients with quantities.\n"
    "  - Pasos: A step-by-step guide on how to prepare the dish.\n"
    "\n"
    "Format your response exactly as follows:\n"
    "Nombre: [Title]\n"
    "Ingredientes: [Ingredient 1] [Ingredient 2]\n"
    "Pasos:\n"
    "1. \n"
    "2. \n"
    "...\n"
    "\n"
    "NOTE: Do not include any explanations, introductions, or extra text—only the complete recipe.\n"
    "Best Answer:"
)


recipe_cultural_adaption_refine_template_str = (
    "Your task is to find the Spanish recipe that is most relevant to the given recipe.\n"
    "Below are some Spanish recipes retrieved through the search:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the source recipe title: {query_str}, select the most relevant recipe from the context.\n"
    "\n"
    "Instructions:\n"
    "- If there are multiple highly relevant recipes, select the one that best aligns with the cultural preferences and norms of Spanish cuisine.\n"
    "- The output must contain a complete recipe, including:\n"
    "  - Nombre: The name of the dish.\n"
    "  - Ingredientes: A detailed list of ingredients with quantities.\n"
    "  - Pasos: A step-by-step guide on how to prepare the dish.\n"
    "\n"
    "Format your response exactly as follows:\n"
    "Nombre: [Title]\n"
    "Ingredientes: [Ingredient 1] [Ingredient 2]\n"
    "Pasos:\n"
    "1. \n"
    "2. \n"
    "...\n"
    "\n"
    "NOTE: Do not include any explanations, introductions, or extra text—only the complete recipe.\n"
    "Best Answer:"
)

def load_prompt(task):
    prompts_dict = {}
    
    if task == "recipe adaption":
        prompts_dict["response_synthesizer:text_qa_template"] = PromptTemplate(recipe_cultural_adaption_select_template_str)
        prompts_dict["response_synthesizer:refine_template"] = PromptTemplate(recipe_cultural_adaption_refine_template_str)
    
    return prompts_dict

