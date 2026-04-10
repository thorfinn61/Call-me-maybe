import argparse
from pathlib import Path
from typing import Any, List
from pydantic import ValidationError
from src.file_handler import load_json
from llm_sdk.llm_sdk import Small_LLM_Model
from src.function_selector import select_function

from src.models import FunctionDefinition, PromptInput

llm = Small_LLM_Model()

def parse_args():
    """Parse les arguments de la ligne de commande pour spécifier les chemins d'entrée/sortie."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="src/data/input/functions_definition.json"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="src/data/input/function_calling_tests.json"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="src/data/output/output.json"
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    raw_functions = load_json(args.functions_definition)
    functions = [FunctionDefinition(**fn) for fn in raw_functions]
    raw_prompts = load_json(args.input)
    prompts = [PromptInput(**inp) for inp in raw_prompts]
    print(f"{len(functions)} fonctions chargées !")
    print(f"{len(prompts)} prompts chargés !")
    for prompt in prompts:
        result = select_function(prompt, functions, llm)
        print(f"Prompt: {prompt.prompt} -> Fonction choisie: {result}")

    
if __name__ == "__main__":
    main()
