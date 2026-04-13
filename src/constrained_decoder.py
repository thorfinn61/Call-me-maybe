import json
import math
import re
from typing import Any

import numpy as np

from src.file_handler import load_json


class ConstrainedDecoder:
    """Construit un JSON valide: {"name": ..., "parameters": {...}}."""

    def __init__(self, model: Any):
        self.model = model
        self.vocab = load_json(self.model.get_path_to_vocab_file()) or {}

        self.quote_id = self.model.encode('"').tolist()[0][0]
        self.comma_id = self.model.encode(",").tolist()[0][0]
        self.close_brace_id = self.model.encode("}").tolist()[0][0]

        self.number_ids = self._collect_number_tokens()
        self.string_ids = self._collect_string_tokens()

        self.pending: list[int] = []
        self.param_keys: list[str] = []
        self.schema: dict[str, Any] = {}
        self.param_index = 0
        self.value_buffer = ""
        self.phase = "name_key"

    def decode(self, prompt_text: str, function_name: str, function_parameters_schema: dict) -> str:
        input_ids = self.model.encode(
            f"Extract parameters into JSON.\nUser: {prompt_text}\nJSON: "
        ).tolist()[0]

        self.param_keys = list(function_parameters_schema.keys())
        self.schema = function_parameters_schema
        self.param_index = 0
        self.value_buffer = ""
        self.pending = []
        self.phase = "name_key"

        generated = ""
        for _ in range(500):
            if not self.pending:
                self._queue_static_tokens(function_name)

            if not self.pending and self.phase == "done":
                break

            logits = self.model.get_logits_from_input_ids(input_ids)

            if self.pending:
                chosen_id = self._pick_forced(logits, self.pending.pop(0))
                text = self.model.decode([chosen_id])
                input_ids.append(chosen_id)
                generated += text
                continue

            param_name = self.param_keys[self.param_index]
            param_type = self._param_type(self.schema[param_name])
            chosen_id = self._pick_dynamic(logits, param_type)
            text = self.model.decode([chosen_id])

            if param_type == "string":
                if chosen_id == self.quote_id:
                    input_ids.append(chosen_id)
                    generated += '"'
                    self._advance_param()
                    continue

                input_ids.append(chosen_id)
                generated += text
                self.value_buffer += text

                # Evite une string infinie si le modele n'emet jamais le guillemet.
                if len(self.value_buffer) >= 256:
                    input_ids.append(self.quote_id)
                    generated += '"'
                    self._advance_param()
                continue

            if param_type == "number":
                if chosen_id in {self.comma_id, self.close_brace_id}:
                    if not self._is_number(self.value_buffer):
                        generated += "0"
                        for zid in self.model.encode("0").tolist()[0]:
                            input_ids.append(zid)
                    self._advance_param()
                    continue

                input_ids.append(chosen_id)
                generated += text
                self.value_buffer += text
                continue

            input_ids.append(chosen_id)
            generated += text

        if self.phase != "done":
            return self._fallback(function_name)

        try:
            json.loads(generated)
            return generated
        except json.JSONDecodeError:
            return self._fallback(function_name)

    def _queue_static_tokens(self, function_name: str) -> None:
        if self.phase == "name_key":
            self.pending = self.model.encode('{"name": "').tolist()[0]
            self.phase = "name_value"
            return

        if self.phase == "name_value":
            self.pending = self.model.encode(f'{function_name}", "parameters": ').tolist()[0]
            self.phase = "params_open"
            return

        if self.phase == "params_open":
            if self.param_keys:
                self.pending = self.model.encode("{").tolist()[0]
                self.phase = "param_key"
            else:
                self.pending = self.model.encode("{}}").tolist()[0]
                self.phase = "done"
            return

        if self.phase == "param_key":
            key = self.param_keys[self.param_index]
            prefix = ", " if self.param_index > 0 else ""
            text = f'{prefix}"{key}": '
            if self._param_type(self.schema[key]) == "string":
                text += '"'
            self.pending = self.model.encode(text).tolist()[0]
            self.value_buffer = ""
            self.phase = "param_value"

    def _pick_forced(self, logits: list[float], token_id: int) -> int:
        masked = [-math.inf] * len(logits)
        if token_id < len(masked):
            masked[token_id] = 0.0
        return int(np.argmax(masked))

    def _pick_dynamic(self, logits: list[float], param_type: str) -> int:
        masked = [-math.inf] * len(logits)

        if param_type == "number":
            allowed = self.number_ids | {self.comma_id, self.close_brace_id}
        else:
            allowed = set(self.string_ids)
            allowed.add(self.quote_id)

        for tid in allowed:
            if tid < len(logits):
                masked[tid] = logits[tid]

        return int(np.argmax(masked))

    def _advance_param(self) -> None:
        self.value_buffer = ""
        self.param_index += 1
        if self.param_index < len(self.param_keys):
            self.phase = "param_key"
        else:
            self.pending = self.model.encode("}}").tolist()[0]
            self.phase = "done"

    def _collect_number_tokens(self) -> set[int]:
        ids: set[int] = set()
        for text, tid in self.vocab.items():
            token = text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else str(text)
            token = token.strip("Ġ▁ ▂▃▄▅▆▇█")
            if token and all(ch in "0123456789.-+eE" for ch in token):
                ids.add(tid)
        return ids

    def _collect_string_tokens(self) -> set[int]:
        ids: set[int] = set()
        for text, tid in self.vocab.items():
            token = text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else str(text)
            if tid == self.quote_id:
                continue
            if '"' in token or "\\" in token or "<|" in token:
                continue
            ids.add(tid)
        return ids

    def _param_type(self, schema_obj: Any) -> str:
        if isinstance(schema_obj, dict):
            return schema_obj.get("type", "string")
        return getattr(schema_obj, "type", "string")

    def _is_number(self, value: str) -> bool:
        value = value.strip()
        if not value:
            return False
        return re.match(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$", value) is not None

    def _fallback(self, function_name: str) -> str:
        params: dict[str, Any] = {}
        for key in self.param_keys:
            params[key] = 0 if self._param_type(self.schema[key]) == "number" else ""
        return json.dumps({"name": function_name, "parameters": params}, ensure_ascii=False)
