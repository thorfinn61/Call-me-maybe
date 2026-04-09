import json
from typing import Dict, List, Any

def load_json(path: str) -> List[Dict[str, Any]]:
	"""Fonction qui charge le fichier json demande"""
	try:
		with open(path) as f:
			data = json.load(f)
		return data
	except json.JSONDecodeError as e:
		print(f"Le fichier JSON est malforme: {e}")
		return []