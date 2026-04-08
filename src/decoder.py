import math
import numpy as np
from enum import Enum, auto
from typing import List, Dict, Any

from src.engine import LLMEngine
from src.data_models import FunctionDefinition

class DecoderState(Enum):
    """Les différents états de notre génération JSON."""
    EXPECTING_BRACE_OPEN = auto()       # Attend '{'
    EXPECTING_NAME_KEY_START = auto()   # Attend '"' (début de la clé)
    EXPECTING_NAME_KEY = auto()         # Attend 'name'
    EXPECTING_NAME_KEY_END = auto()     # Attend '"' (fin de la clé)
    EXPECTING_NAME_COLON = auto()       # Attend ':'
    EXPECTING_FUNCTION_QUOTE = auto()   # Attend '"' avant le nom de fonction
    EXPECTING_FUNCTION_NAME_1 = auto()  # Attend 'fn'
    EXPECTING_FUNCTION_NAME_2 = auto()  # Attend '_add'
    EXPECTING_FUNCTION_NAME_3 = auto()  # Attend '_numbers'
    EXPECTING_FUNCTION_END_QUOTE = auto() # Attend '"' après le nom
    EXPECTING_COMMA = auto()            # Attend ','
    EXPECTING_PARAMS_KEY_START = auto() # Attend '"' (début de la clé parameters)
    EXPECTING_PARAMS_KEY = auto()       # Attend 'parameters'
    EXPECTING_PARAMS_KEY_END = auto()   # Attend '"' (fin de la clé parameters)
    EXPECTING_PARAMS_COLON = auto()     # Attend ':'
    EXPECTING_PARAMS_DICT = auto()      # Attend '{'
    DONE = auto()                       # JSON terminé !

def mask_logits(logits: List[float], allowed_token_ids: List[int]) -> np.ndarray:
    """
    Prend les logits bruts de l'IA et met à -infini tous les tokens qui 
    ne sont pas dans allowed_token_ids.
    """
    # 1. Convertir la liste de logits en tableau NumPy
    logits_array = np.array(logits)
    
    # 2. Créer un tableau de la même taille rempli de -infini
    masked_logits = np.full_like(logits_array, -math.inf)
    
    # 3. Restaurer UNIQUEMENT les scores des tokens autorisés
    if allowed_token_ids:
        masked_logits[allowed_token_ids] = logits_array[allowed_token_ids]
    else:
        # Cas d'urgence (sécurité) si aucun token n'est autorisé
        pass us les tokens entiers qui com
        
    return masked_logits

class JSONDecoder:
    """La machine à états qui va contraindre la génération."""
    def __init__(self, engine: LLMEngine, functions: List[FunctionDefinition]):
        self.engine = engine
        self.functions = functions
        self.prefix_trie = engine.build_prefix_trie()
        
    def get_allowed_tokens_for_state(self, state: DecoderState) -> List[int]:
        """Retourne la liste des IDs de tokens autorisés selon l'état actuel."""
        allowed = []
        if state == DecoderState.EXPECTING_BRACE_OPEN:
            allowed = self.prefix_trie.get('{', []) + self.prefix_trie.get(' {', [])
        
        elif state == DecoderState.EXPECTING_NAME_KEY_START:
            allowed = self.prefix_trie.get('"', []) + self.prefix_trie.get(' "', [])
            
        elif state == DecoderState.EXPECTING_NAME_KEY:
            allowed = self.prefix_trie.get('name', []) + self.prefix_trie.get(' name', [])
            
        elif state == DecoderState.EXPECTING_NAME_KEY_END:
            allowed = self.prefix_trie.get('"', []) + self.prefix_trie.get(' "', [])
            
        elif state == DecoderState.EXPECTING_NAME_COLON:
            allowed = self.prefix_trie.get(':', []) + self.prefix_trie.get(' :', [])
            
        elif state == DecoderState.EXPECTING_FUNCTION_QUOTE:
            allowed = self.prefix_trie.get('"', []) + self.prefix_trie.get(' "', [])
            
        elif state == DecoderState.EXPECTING_FUNCTION_NAME_1:
            allowed = self.prefix_trie.get('fn', []) + self.prefix_trie.get(' fn', [])
            
        elif state == DecoderState.EXPECTING_FUNCTION_NAME_2:
            allowed = self.prefix_trie.get('_add', []) + self.prefix_trie.get(' _add', [])
            
        elif state == DecoderState.EXPECTING_FUNCTION_NAME_3:
            allowed = self.prefix_trie.get('_numbers', []) + self.prefix_trie.get(' _numbers', [])
                
        elif state == DecoderState.EXPECTING_FUNCTION_END_QUOTE:
            allowed = self.prefix_trie.get('"', []) + self.prefix_trie.get(' "', [])
            
        elif state == DecoderState.EXPECTING_COMMA:
            allowed = self.prefix_trie.get(',', []) + self.prefix_trie.get(' ,', [])
            
        elif state == DecoderState.EXPECTING_PARAMS_KEY_START:
            allowed = self.prefix_trie.get('"', []) + self.prefix_trie.get(' "', [])
            
        elif state == DecoderState.EXPECTING_PARAMS_KEY:
            allowed = self.prefix_trie.get('parameters', []) + self.prefix_trie.get(' parameters', [])
            
        elif state == DecoderState.EXPECTING_PARAMS_KEY_END:
            allowed = self.prefix_trie.get('"', []) + self.prefix_trie.get(' "', [])
            
        elif state == DecoderState.EXPECTING_PARAMS_COLON:
            allowed = self.prefix_trie.get(':', []) + self.prefix_trie.get(' :', [])
            
        elif state == DecoderState.EXPECTING_PARAMS_DICT:
            allowed = self.prefix_trie.get(' {', []) + self.prefix_trie.get('{', [])
        
        return list(set(allowed)) # Déduplication des IDs

    def generate_function_call(self, prompt: str) -> Dict[str, Any]:
        """
        La fameuse boucle de génération token par token.
        """
        state = DecoderState.EXPECTING_BRACE_OPEN
        
        # On encode le prompt avec le formatage attendu par l'IA
        formatted_prompt = f"User: {prompt}\nCall: "
        
        # Le SDK renvoie un Tensor 2D, on le convertit en liste 1D
        input_tensor = self.engine.model.encode(formatted_prompt)
        current_input_ids = input_tensor[0].tolist()
        
        generated_token_ids = []
        max_tokens = 30 # Sécurité absolue pour éviter une boucle infinie
        
        print(f"\n[Génération pour : '{prompt}']")
        
        for step in range(max_tokens):
            if state == DecoderState.DONE:
                break
                
            # 1. Obtenir les logits pour le prochain token
            logits = self.engine.model.get_logits_from_input_ids(current_input_ids)
            
            # 2. Déterminer quels tokens sont autorisés pour l'état actuel
            allowed_ids = self.get_allowed_tokens_for_state(state)
            
            # 3. Appliquer le masque (-infini sur les tokens non autorisés)
            masked_logits = mask_logits(logits, allowed_ids)
            
            # 4. Sélectionner le token avec le plus grand score (argmax)
            best_token_id = int(np.argmax(masked_logits))
            
            # 5. Ajouter le token choisi à la séquence
            current_input_ids.append(best_token_id)
            generated_token_ids.append(best_token_id)
            
            # 6. Décoder temporairement le texte TOKÉNISE
            full_text = self.engine.model.decode(generated_token_ids)
            token_text = self.engine.model.decode([best_token_id])
            print(f"Step {step} | State: {state.name:<30} | Texte complet: '{full_text}'")
            
            # 7. Transition vers le prochain état strict (basé sur une analyse propre)
            # Puisqu'on autorise qu'un seul token (ou un groupe ultra restreint), on 
            # contrôle exactement le déroulé et les transitions sans erreur !
            
            if state == DecoderState.EXPECTING_BRACE_OPEN:
                state = DecoderState.EXPECTING_NAME_KEY_START
                
            elif state == DecoderState.EXPECTING_NAME_KEY_START:
                state = DecoderState.EXPECTING_NAME_KEY
                
            elif state == DecoderState.EXPECTING_NAME_KEY:
                state = DecoderState.EXPECTING_NAME_KEY_END
                
            elif state == DecoderState.EXPECTING_NAME_KEY_END:
                state = DecoderState.EXPECTING_NAME_COLON
                
            elif state == DecoderState.EXPECTING_NAME_COLON:
                state = DecoderState.EXPECTING_FUNCTION_QUOTE
                
            elif state == DecoderState.EXPECTING_FUNCTION_QUOTE:
                state = DecoderState.EXPECTING_FUNCTION_NAME_1
                
            elif state == DecoderState.EXPECTING_FUNCTION_NAME_1:
                state = DecoderState.EXPECTING_FUNCTION_NAME_2
                
            elif state == DecoderState.EXPECTING_FUNCTION_NAME_2:
                state = DecoderState.EXPECTING_FUNCTION_NAME_3
                
            elif state == DecoderState.EXPECTING_FUNCTION_NAME_3:
                state = DecoderState.EXPECTING_FUNCTION_END_QUOTE
                
            elif state == DecoderState.EXPECTING_FUNCTION_END_QUOTE:
                state = DecoderState.EXPECTING_COMMA
                
            elif state == DecoderState.EXPECTING_COMMA:
                state = DecoderState.EXPECTING_PARAMS_KEY_START
                
            elif state == DecoderState.EXPECTING_PARAMS_KEY_START:
                state = DecoderState.EXPECTING_PARAMS_KEY
                
            elif state == DecoderState.EXPECTING_PARAMS_KEY:
                state = DecoderState.EXPECTING_PARAMS_KEY_END
                
            elif state == DecoderState.EXPECTING_PARAMS_KEY_END:
                state = DecoderState.EXPECTING_PARAMS_COLON
                
            elif state == DecoderState.EXPECTING_PARAMS_COLON:
                state = DecoderState.EXPECTING_PARAMS_DICT
                
            elif state == DecoderState.EXPECTING_PARAMS_DICT:
                state = DecoderState.DONE
        
        # Décodage complet du json généré
        final_text = self.engine.model.decode(generated_token_ids)
        print("\n--- TEXTE FINAL GÉNÉRÉ ---")
        print(final_text)
        print("----------------------------\n")
        
        return {}  # Renverra le dictionnaire Python final plus tard
