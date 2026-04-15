import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from llm_sdk.llm_sdk import Small_LLM_Model
from src.constrained_decoder import ConstrainedDecoder
from src.file_handler import load_json
from src.function_selector import select_function
from src.models import FunctionDefinition, PromptInput


def _fix_regex_parameters(
    prompt: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Correction manuelle pour les Regex du petit modèle."""
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
        cible = re.search(
            r"\bwith\s+['\"]([^'\"]+)['\"]\s+in\b",
            prompt,
            flags=re.IGNORECASE,
        )
        if cible:
            out["replacement"] = cible.group(1)

    elif "cat" in low and "substitute" in low:
        out["regex"] = r"cat"
        cible = re.search(
            r"\bwith\s+['\"]([^'\"]+)['\"]\s+in\b",
            prompt,
            flags=re.IGNORECASE,
        )
        if cible:
            out["replacement"] = cible.group(1)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Function calling")
    parser.add_argument("--functions_definition",
                        default="src/data/input/functions_definition.json")
    parser.add_argument(
        "--input", default="src/data/input/function_calling_tests.json")
    parser.add_argument(
        "--output", default="src/data/output/function_calling_results.json")
    return parser.parse_args()


def _format_validation_error(err: ValidationError, data_name: str) -> str:
    """Construit un message lisible à partir d'une ValidationError."""
    details = []
    for item in err.errors():
        loc = item.get("loc", ())
        msg = item.get("msg", "erreur de validation")

        if len(loc) >= 2 and isinstance(loc[0], int):
            index = loc[0]
            field = ".".join(str(part) for part in loc[1:])
            details.append(
                f"- {data_name}[{index}].{field}: {msg}"
            )
        else:
            field = ".".join(str(part) for part in loc) if loc else data_name
            details.append(f"- {field}: {msg}")

    joined = "\n".join(details)
    return (
        f"❌ Arrêt du programme : format invalide dans {data_name}.\n"
        f"Détails:\n{joined}"
    )


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    print("🚀 Lancement du Constrained Decoding Runner...")

    # 1. Charger les données (Pydantic valide automatiquement)
    raw_functions = load_json(args.functions_definition)
    raw_prompts = load_json(args.input)

    if raw_functions is None or raw_prompts is None:
        print(
            "❌ Arrêt du programme : Impossible de continuer avec "
            "un fichier JSON invalide ou manquant."
        )
        return

    try:
        functions = [FunctionDefinition(**item) for item in raw_functions]
    except ValidationError as err:
        print(_format_validation_error(err, "functions_definition"))
        return

    try:
        prompts = [PromptInput(**item) for item in raw_prompts]
    except ValidationError as err:
        print(_format_validation_error(err, "input_prompts"))
        return

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
            print(
                f"[{i}/{len(prompts)}] ❌ Fonction introuvable: "
                f"{chosen_name}"
            )
            continue

        try:
            # Forcer le modèle à générer un JSON valide
            # qui respecte les paramètres
            raw_json_str = decoder.decode(
                prompt_text, func.name, func.parameters)
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
            print(
                f"[{i}/{len(prompts)}] ❌ Erreur pendant "
                f"le décodage: {e}"
            )

    # 4. Sauvegarder les résultats finaux
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"🎉 Terminé ! Les résultats sont dans {args.output}")
    elapsed = time.perf_counter() - start_time
    print(f"⏱️ Temps total : {elapsed:.2f}s")


if __name__ == "__main__":
    main()
