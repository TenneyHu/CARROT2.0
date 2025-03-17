from llama_index.core import PromptTemplate

recipe_cultural_adaption_select_template_str = (
    "Tu tarea es encontrar la receta española que sea más relevante para la receta dada.\n"
    "A continuación, se presentan algunas recetas españolas recuperadas a través de la búsqueda:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dada la receta de origen titulada: {query_str}, selecciona la receta más relevante del contexto.\n"
    "\n"
    "Instrucciones:\n"
    "- Si hay varias recetas muy relevantes, selecciona la que mejor se alinee con las preferencias y convenciones culturales de la cocina española.\n"
    "- La respuesta debe contener una receta completa, incluyendo:\n"
    " - Nombre: El nombre del plato.\n"
    " - Ingredientes: Una lista detallada de ingredientes con cantidades.\n"
    " - Pasos: Una guía paso a paso sobre cómo preparar el plato.\n"
    "\n"
    "Formato de la respuesta exactamente como sigue:\n"
    "Nombre: [Título]\n"
    "Ingredientes: [Ingrediente 1] [Ingrediente 2]\n"
    "Pasos:\n"
    "1. \n"
    "2. \n"
    "...\n"
    "\n"
    "NOTA: No incluyas explicaciones, introducciones ni texto adicional—únicamente la receta completa.\n"
    "Mejor respuesta:"
)


recipe_cultural_adaption_refine_template_str = (
    "Tu tarea es encontrar la receta española que sea más relevante para la receta dada.\n"
    "A continuación, se presentan algunas recetas españolas recuperadas a través de la búsqueda:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dada la receta de origen titulada: {query_str}, selecciona la receta más relevante del contexto.\n"
    "\n"
    "Instrucciones:\n"
    "- Si hay varias recetas muy relevantes, selecciona la que mejor se alinee con las preferencias y convenciones culturales de la cocina española.\n"
    "- La respuesta debe contener una receta completa, incluyendo:\n"
    " - Nombre: El nombre del plato.\n"
    " - Ingredientes: Una lista detallada de ingredientes con cantidades.\n"
    " - Pasos: Una guía paso a paso sobre cómo preparar el plato.\n"
    "\n"
    "Formato de la respuesta exactamente como sigue:\n"
    "Nombre: [Título]\n"
    "Ingredientes: [Ingrediente 1] [Ingrediente 2]\n"
    "Pasos:\n"
    "1. \n"
    "2. \n"
    "...\n"
    "\n"
    "NOTA: No incluyas explicaciones, introducciones ni texto adicional—únicamente la receta completa.\n"
    "Mejor respuesta:"
)

def load_prompt(task):
    prompts_dict = {}
    
    if task == "recipe adaption":
        prompts_dict["response_synthesizer:text_qa_template"] = PromptTemplate(recipe_cultural_adaption_select_template_str)
        prompts_dict["response_synthesizer:refine_template"] = PromptTemplate(recipe_cultural_adaption_refine_template_str)
    
    return prompts_dict

