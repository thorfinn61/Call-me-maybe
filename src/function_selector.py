import numpy as np
from typing import List
from src.models import FunctionDefinition, PromptInput

def select_function(prompt: PromptInput, functions: List[FunctionDefinition], model) -> str:
    menu_texte = "Here are the tools available:\n"
    for func in functions:
        menu_texte += f"- {func.name}: {func.description}\n"
    
    instruction = (
        "You are an assistant. You must choose the correct function name based on the user's request.\n"
        "Answer strictly and ONLY with the name of the function. No other text.\n\n"
    )
    
    user_request = f"User Request: {prompt.prompt}\n\n"
    
    final_prompt = instruction + menu_texte + "\n" + user_request + "Function to call:"
    input_ids = model.encode(final_prompt).tolist()[0]
    
    generated_ids = []
    
    for _ in range(20):
        logits = model.get_logits_from_input_ids(input_ids)
        predicted_id = int(np.argmax(logits))
        
        generated_ids.append(predicted_id)
        input_ids.append(predicted_id)
        
    generated_text = model.decode(generated_ids)
    
    for func in functions:
        if func.name in generated_text:
            return func.name

    return generated_text.strip()
