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

def main() -> None:
    args = parse_args()

    print(f"ex: {args}")

if __name__ == "__main__":
    main()
