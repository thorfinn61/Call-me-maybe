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

SUPPORTED_PARAM_TYPES = {"string", "number"}

EXPECTED_FUNCTION_SIGNATURES: dict[str, dict[str, str]] = {
    "fn_add_numbers": {"a": "number", "b": "number"},
    "fn_greet": {"name": "string"},
    "fn_reverse_string": {"s": "string"},
    "fn_get_square_root": {"a": "number"},
    "fn_substitute_string_with_regex": {
        "source_string": "string",
        "regex": "string",
        "replacement": "string",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding"
    )
    parser.add_argument(
        "--functions_definition",
        default="src/data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="src/data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="src/data/output/function_calling_results.json",
    )
    return parser.parse_args()


def _schema_type(field: Any) -> str:
    if isinstance(field, dict):
        return str(field.get("type", "string"))
    return str(getattr(field, "type", "string"))


def _extract_quoted(text: str) -> list[str]:
    chunks = [m.group(1) for m in re.finditer(r'"([^"]+)"', text)]
    chunks += [
        m.group(1) for m in re.finditer(r"(?<!\w)'([^']+)'(?!\w)", text)
    ]
    return chunks


def _normalize_regex_call(
    prompt: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    out = dict(params)
    quoted = _extract_quoted(prompt)
    if quoted:
        out["source_string"] = max(quoted, key=len)

    low = prompt.lower()
    if "numbers" in low:
        out["regex"] = r"[0-9]+"
    elif "vowels" in low:
        out["regex"] = r"[aeiouAEIOU]"
    elif "cat" in low and "substitute" in low:
        out["regex"] = r"cat"

    repl = re.search(
        r"\bwith\s+['\"]([^'\"]+)['\"]\s+in\b",
        prompt,
        flags=re.IGNORECASE,
    )
    if repl:
        out["replacement"] = repl.group(1)
    elif "numbers" in low:
        tail = re.search(r"\bwith\s+([^\s]+)", prompt, flags=re.IGNORECASE)
        if tail:
            out["replacement"] = tail.group(1).strip("'\"")

    return out


def _load_data(
    functions_path: str,
    prompts_path: str,
) -> tuple[list[FunctionDefinition], list[PromptInput]]:
    raw_functions = load_json(functions_path)
    raw_prompts = load_json(prompts_path)

    if raw_functions is None:
        raise ValueError(
            "echec de lecture de --functions_definition "
            "(fichier manquant ou JSON invalide)"
        )
    if raw_prompts is None:
        raise ValueError(
            "echec de lecture de --input "
            "(fichier manquant ou JSON invalide)"
        )
    if not isinstance(raw_functions, list):
        raise ValueError("--functions_definition doit contenir une liste JSON")
    if not isinstance(raw_prompts, list):
        raise ValueError("--input doit contenir une liste JSON")

    try:
        functions = [FunctionDefinition(**item) for item in raw_functions]
        prompts = [PromptInput(**item) for item in raw_prompts]
    except ValidationError as exc:
        raise ValueError(f"donnees invalides: {exc}") from exc

    for fn in functions:
        for param_name, field in fn.parameters.items():
            field_type = _schema_type(field)
            if field_type not in SUPPORTED_PARAM_TYPES:
                raise ValueError(
                    "type non supporte dans --functions_definition: "
                    f"{fn.name}.{param_name}={field_type} "
                    "(types autorises: "
                    f"{', '.join(sorted(SUPPORTED_PARAM_TYPES))})"
                )

    _validate_function_signatures(functions)
    return functions, prompts


def _validate_function_signatures(functions: list[FunctionDefinition]) -> None:
    by_name = {fn.name: fn for fn in functions}

    missing = sorted(set(EXPECTED_FUNCTION_SIGNATURES) - set(by_name))
    unexpected = sorted(set(by_name) - set(EXPECTED_FUNCTION_SIGNATURES))

    if missing:
        raise ValueError(
            "fonctions manquantes dans --functions_definition: "
            f"{', '.join(missing)}"
        )
    if unexpected:
        raise ValueError(
            "fonctions inattendues dans --functions_definition: "
            f"{', '.join(unexpected)}"
        )

    for function_name, expected_params in EXPECTED_FUNCTION_SIGNATURES.items():
        fn = by_name[function_name]
        actual = {
            key: _schema_type(field)
            for key, field in fn.parameters.items()
        }

        missing_params = sorted(set(expected_params) - set(actual))
        extra_params = sorted(set(actual) - set(expected_params))

        if missing_params:
            raise ValueError(
                f"parametres manquants pour {function_name}: "
                f"{', '.join(missing_params)}"
            )
        if extra_params:
            raise ValueError(
                f"parametres inattendus pour {function_name}: "
                f"{', '.join(extra_params)}"
            )

        for param_name, expected_type in expected_params.items():
            actual_type = actual[param_name]
            if actual_type != expected_type:
                raise ValueError(
                    f"type invalide pour {function_name}.{param_name}: "
                    f"attendu={expected_type}, recu={actual_type}"
                )


def _valid_item(
    prompt: str,
    name: Any,
    params: Any,
    schema: dict[str, Any],
) -> tuple[bool, str]:
    if not isinstance(prompt, str):
        return False, "'prompt' doit etre une string"
    if not isinstance(name, str) or not name:
        return False, "'name' doit etre une string non vide"
    if not isinstance(params, dict):
        return False, "'parameters' doit etre un objet JSON"

    for key, field in schema.items():
        if key not in params:
            return False, f"parametre manquant: {key}"

        expected = _schema_type(field)
        value = params[key]
        if expected == "string" and not isinstance(value, str):
            return False, f"type invalide pour '{key}'"
        if expected == "number" and not (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            return False, f"type invalide pour '{key}'"

    return True, ""


def main() -> None:
    start = time.perf_counter()
    args = parse_args()

    print("Constrained Decoding Runner")
    print(f"Input prompts : {args.input}")
    print(f"Functions     : {args.functions_definition}")
    print(f"Output        : {args.output}")

    try:
        functions, prompts = _load_data(
            args.functions_definition,
            args.input,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return

    model = Small_LLM_Model()
    decoder = ConstrainedDecoder(model)
    output: list[dict[str, Any]] = []

    print(f"[LOAD] {len(functions)} fonctions chargees")
    print(f"[LOAD] {len(prompts)} prompts charges")

    for i, prompt in enumerate(prompts, 1):
        chosen_name = select_function(prompt, functions, model)
        func = next((f for f in functions if f.name == chosen_name), None)

        if func is None:
            print(
                f"[{i}/{len(prompts)}] FAILED "
                f"fonction introuvable: {chosen_name}"
            )
            continue

        try:
            raw = decoder.decode(prompt.prompt, func.name, func.parameters)
        except ValueError as exc:
            print(f"[{i}/{len(prompts)}] FAILED {exc}")
            continue

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[{i}/{len(prompts)}] FAILED JSON invalide")
            continue

        name = parsed.get("name")
        params = parsed.get("parameters", {})
        if name == "fn_substitute_string_with_regex":
            params = _normalize_regex_call(prompt.prompt, params)

        ok, reason = _valid_item(prompt.prompt, name, params, func.parameters)
        if not ok:
            print(f"[{i}/{len(prompts)}] FAILED {reason}")
            continue

        output.append(
            {"prompt": prompt.prompt, "name": name, "parameters": params}
        )
        print(f"[{i}/{len(prompts)}] OK {name}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Sortie : {args.output}")
    print(f"Temps total : {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
