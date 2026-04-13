import json
from typing import Any


def load_json(path: str) -> Any:
    """Charge un JSON et retourne son contenu, ou None en cas d'erreur."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Fichier introuvable: {path}")
    except json.JSONDecodeError as e:
        print(f"Le fichier JSON est malforme ({path}): {e}")
    except OSError as e:
        print(f"Impossible de lire le fichier {path}: {e}")
    return None
