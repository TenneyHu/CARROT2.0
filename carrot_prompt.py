from llama_index.core import PromptTemplate

recipe_cultural_adaption_template_str = (
    "Convert the following recipe into a Spanish recipe so that it fits Spanish culture, is consistent with Spanish culinary knowledge, and aligns with the style of Spanish recipes and the availability of ingredients.\n"
    "Below are some relevant Spanish recipes retrieved through search, which may be helpful for the task:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the source recipe {query_str}, Use these retrieved recipes to adapt it into a Spanish recipe.\n"
    "\n"
    "Instructions:\n"
    "Looking for relevant recipes among the retrieved ones to use as references. If there are multiple highly relevant recipes, select the one that best aligns with the cultural preferences and norms of Spanish cuisine.\n"
    "The output recipe should be complete, including detailed ingredients and step-by-step instructions. You can refer to the style of the retrieved Spanish recipes.\n"
    "Format your response exactly as follows:\n"
    "Nombre: [Title]\n"
    "Ingredientes: [Ingredient 1] [Ingredient 2]\n"
    "Pasos:\n"
    "1. \n"
    "2. \n"
    "...\n"
    "\n"
    "Best Answer:"
)

recipe_cultural_adaption_spansih_template_str = (
    "Convierte la siguiente receta en una receta española para que se adapte a la cultura española, sea coherente con el conocimiento culinario español y se alinee con el estilo de las recetas españolas y la disponibilidad de ingredientes.\n"
    "A continuación se muestran algunas recetas españolas relevantes recuperadas mediante búsqueda, que pueden ser útiles para la tarea:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dada la receta original {query_str}, utiliza las siguientes recetas recuperadas para adaptarla a una receta española.\n"
    "\n"
    "Instrucciones:\n"
    "Busca recetas relevantes entre las recuperadas para usarlas como referencia. Si hay varias recetas muy relevantes, selecciona la que mejor se alinee con las preferencias culturales y convenciones de la cocina española.\n"
    "La receta resultante debe estar completa, incluyendo ingredientes detallados e instrucciones paso a paso. Puedes guiarte por el estilo de las recetas españolas recuperadas.\n"
    "Da formato a tu respuesta exactamente de la siguiente manera:\n"
    "Nombre: [Título]\n"
    "Ingredientes: [Ingrediente 1] [Ingrediente 2]\n"
    "Pasos:\n"
    "1. \n"
    "2. \n"
    "...\n"
    "\n"
    "Mejor respuesta:"
)

def load_prompt(task):
    if task == "recipe adaption":
        # return PromptTemplate(recipe_cultural_adaption_template_str)
        return PromptTemplate(recipe_cultural_adaption_spansih_template_str)
    
    return PromptTemplate("")

