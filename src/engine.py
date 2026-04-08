import json
import numpy as np
from pathlib import Path
from typing import Dict

from llm_sdk import Small_LLM_Model

class LLMEngine:
    """Gère les interactions de bas niveau avec le modèle de langage."""
    
    def __init__(self) -> None:
        print("Chargement du modèle Qwen... (patientez)")
        self.model = Small_LLM_Model()
        self.vocab = self._load_vocabulary()

    def _load_vocabulary(self) -> Dict[str, int]:
        """Charge le fichier JSON du vocabulaire en mémoire."""
        vocab_path = Path(self.model.get_path_to_vocab_file())
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
            
        print(f"Vocabulaire chargé : {len(raw_vocab)} tokens connus.")
        return raw_vocab

    def build_prefix_trie(self) -> Dict[str, list[int]]:
        """Construit un dictionnaire qui associe un mot exact à la liste de ses tokens (IDs)."""
        prefix_map = {}
        for token_text, token_id in self.vocab.items():
            clean_text = token_text.replace('Ġ', ' ')
            
            # On stocke le token entier pour avoir une correspondance exacte
            if clean_text not in prefix_map:
                prefix_map[clean_text] = []
            prefix_map[clean_text].append(token_id)
            
        return prefix_map

    def test_prediction(self, test_prompt: str) -> None:
        """Fonction temporaire pour tester la génération du prochain token."""
        print(f"\n--- TEST DU MODELE ---")
        print(f"Prompt brut: '{test_prompt}'")
        
        # 1. Encodage du texte (le SDK renvoie un tenseur PyTorch 2D)
        input_tensor = self.model.encode(test_prompt)
        
        # Le SDK demande une liste d'entiers 1D pour get_logits_from_input_ids
        input_ids_list = input_tensor[0].tolist()
        
        # 2. Inférence (prédiction) des logits
        # (le SDK renvoie une liste de floats pour le CA prochain token)
        logits = self.model.get_logits_from_input_ids(input_ids_list)
        
        # 3. Trouver le token avec le meilleur score (argmax)
        best_token_id = int(np.argmax(logits))
        
        # 4. Décoder pour voir le texte généré (le SDK attend une liste)
        generated_text = self.model.decode([best_token_id])
        
        print(f"Meilleur logit pour le token ID : {best_token_id}")
        print(f"Ce qui correspond au texte : '{generated_text}'")
        print(f"----------------------\n")
        
        # Test supplémentaire de la structure optimisée :
        prefix_dict = self.build_prefix_trie()
        print(f"Si l'on cherche les IDs pour le caractère '{{', il en existe {len(prefix_dict.get('{', []))} dans le vocabulaire.")
        print(f"ID pour le mot 'name' : {prefix_dict.get('name', [])[:10]} (limité à 10)")
        print(f"----------------------\n")
