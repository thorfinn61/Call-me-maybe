import numpy as np
from typing import Any, List
from src.models import FunctionDefinition, PromptInput


def _matches_prefix(sequence: list[int], prefix: list[int]) -> bool:
    if len(prefix) > len(sequence):
        return False
    return sequence[: len(prefix)] == prefix


def select_function(
    prompt: PromptInput,
    functions: List[FunctionDefinition],
    model: Any,
) -> str:
    """Fonction qui rend le nom de fonction correspondant au prompt"""
    menu_texte = "Here are the tools available:\n"
    for func in functions:
        menu_texte += f"- {func.name}: {func.description}\n"

    instruction = (
        "You are an assistant. "
        "You must choose the correct function name "
        "based on the user's request.\n"
        "Answer strictly and ONLY with the name of the function. "
        "No other text.\n\n"
    )

    user_request = f"User Request: {prompt.prompt}\n\n"

    final_prompt = (
        instruction
        + menu_texte
        + "\n"
        + user_request
        + "Function to call:"
    )
    input_ids = model.encode(final_prompt).tolist()[0]

    # Autorise deux variantes: tokenisation avec ou sans espace initial.
    candidates: list[tuple[str, list[int]]] = []
    for func in functions:
        for variant in (func.name, f" {func.name}"):
            token_ids = model.encode(variant).tolist()[0]
            if token_ids:
                candidates.append((func.name, token_ids))

    generated_ids: list[int] = []
    max_len = max(len(seq) for _, seq in candidates)

    while len(generated_ids) < max_len:
        matching = [
            (name, seq)
            for name, seq in candidates
            if _matches_prefix(seq, generated_ids)
        ]
        if not matching:
            break

        completed = [
            name for name, seq in matching if len(seq) == len(generated_ids)
        ]
        next_token_ids = {
            seq[len(generated_ids)]
            for _, seq in matching
            if len(seq) > len(generated_ids)
        }

        if completed and not next_token_ids:
            break

        logits = model.get_logits_from_input_ids(input_ids)
        masked = [-np.inf] * len(logits)
        for token_id in next_token_ids:
            if token_id < len(logits):
                masked[token_id] = logits[token_id]

        predicted_id = int(np.argmax(masked))
        generated_ids.append(predicted_id)
        input_ids.append(predicted_id)

    generated_text = str(model.decode(generated_ids)).strip()

    for func in functions:
        if generated_text == func.name:
            return func.name

    for func in functions:
        if func.name in generated_text:
            return func.name

    return generated_text
