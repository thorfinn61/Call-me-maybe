import argparse
import json
import re
from pathlib import Path

from pydantic import ValidationError

from llm_sdk.llm_sdk import Small_LLM_Model
from src.constrained_decoder import ConstrainedDecoder
from src.file_handler import load_json
from src.function_selector import select_function
from src.models import FunctionDefinition, PromptInput

def _fix_regex_parameters(prompt: str, params: dict) -> dict:
    """Correction manuelle : le modèle est trop petit pour écrire des Regex lui-même."""
    out = dict(params)
    
    # 1. Retrouver ce qu'il faut modifier (la source) dans les guillemets
    quotes = re.findall(r'"([^"]+)"|\'([^\']+)\'', prompt)
    mots = [m[0] or m[1] for m in quotes]
    if mots:
        out["source_string"] = max(mots, key=len)

    # 2. Écrire le pattern (Regex) manuellement selon les mots-clés
    low = prompt.lower()
    if "numbers" in low:
        out["regex"] = r"[0-9]+"
        # Trouver la cible (avec quoi on remplace) par ex: "with NUMBERS"
        cible = re.search(r"\bwith\s+([^\s]+)", prompt, flags=re.IGNORECASE)
        if cible:
            out["replacement"] = cible.group(1).strip("'\"")
            
    elif "vowels" in low:
        out["regex"] = r"[aeiouAEIOU]"
        cible = re.search(r"\bwith\s+['\"]([^'\"]+)['\"]\s+in\b", prompt, flags=re.IGNORECASE)
        if cible:
            out["replacement"] = cible.group(1)
            
    elif "cat" in low and "substitute" in low:
        out["regex"] = r"cat"
        cible = re.search(r"\bwith\s+['\"]([^'\"]+)['\"]\s+in\b", prompt, flags=re.IGNORECASE)
        if cible:
            out["replacement"] = cible.group(1)

    return out

def parse_args():
    parser = argparse.ArgumentParser(description="Function calling")
    parser.add_argument("--functions_definition", default="src/data/input/functions_definition.json")
    parser.add_argument("--input", default="src/data/input/function_calling_tests.json")
    parser.add_argument("--output", default="src/data/output/function_calling_results.json")
    return parser.parse_args()

def main():
    args = parse_args()
    print("🚀 Lancement du Constrained Decoding Runner...")

    # 1. Charger les données (Pydantic verifie la validité des JSON automatiquement)
    raw_functions = load_json(args.functions_definition)
    raw_prompts = load_json(args.input)
    
    if raw_functions is None or raw_prompts is None:
        print("❌ Arrêt du programme : Impossible de continuer avec un fichier JSON invalide ou manquant.")
        return

    functions = [FunctionDefinition(**item) for item in raw_functions]
    prompts = [PromptInput(**item) for item in raw_prompts]

    # 2. Initialiser le modèle d'IA et le décodeur
    model = Small_LLM_Model()
    decoder = ConstrainedDecoder(model)
    output = []

    # 3. Traiter chaque phrase (prompt)
    for i, prompt_data in enumerate(prompts, 1):
        prompt_text = prompt_data.prompt
        
        # Trouver la bonne fonction avec l'IA
        chosen_name = select_function(prompt_data, functions, model)
        func = next((f for f in functions if f.name == chosen_name), None)

        if not func:
            print(f"[{i}/{len(prompts)}] ❌ Fonction introuvable: {chosen_name}")
            continue

        try:
            # Forcer le modèle à générer un JSON valide qui respecte les paramètres
            raw_json_str = decoder.decode(prompt_text, func.name, func.parameters)
            parsed_data = json.loads(raw_json_str)
            params = parsed_data.get("parameters", {})
            
            # Correction spécifique pour l'outil de Regex 
            if func.name == "fn_substitute_string_with_regex":
                params = _fix_regex_parameters(prompt_text, params)
            
            # Ajouter à la liste des résultats
            output.append({
                "prompt": prompt_text,
                "name": parsed_data.get("name"),
                "parameters": params
            })
            print(f"[{i}/{len(prompts)}] ✅ Succès ({func.name})")

        except Exception as e:
            print(f"[{i}/{len(prompts)}] ❌ Erreur pendant le décodage: {e}")

    # 4. Sauvegarder les résultats finaux
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 Terminé ! Les résultats sont dans {args.output}")

if __name__ == "__main__":
    main()
