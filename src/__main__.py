import argparse
import json
from pathlib import Path
from typing import Any, List
from pydantic import ValidationError

from src.data_models import FunctionDefinition, PromptInput

def parse_args():
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

def load_json_file(file_path: str) -> List[Any]:
    path = Path(file_path)
    if not path.is_file():
        print(f"Erreur: Le fichier '{file_path}' est introuvable.")
        return []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Erreur de syntaxe JSON dans '{file_path}': {e}")
        return []
    except Exception as e:
        print(f"Erreur inattendue lors de la lecture de '{file_path}': {e}")
        return []

def main() -> None:
    args = parse_args()

    raw_functions = load_json_file(args.functions_definition)
    raw_prompts = load_json_file(args.input)

    functions: List[FunctionDefinition] = []
    for item in raw_functions:
        try:
            func_def = FunctionDefinition(**item)
            functions.append(func_def)
        except ValidationError as e:
            print(f"Avertissement: Fonction ignorée car invalide: {e}")

    prompts: List[PromptInput] = []
    for item in raw_prompts:
        try:
            prompt_input = PromptInput(**item)
            prompts.append(prompt_input)
        except ValidationError as e:
            print(f"Avertissement: Prompt ignoré car invalide: {e}")

    print(f"Succès: {len(functions)} fonctions et {len(prompts)} prompts chargés.")

if __name__ == "__main__":
    main()
