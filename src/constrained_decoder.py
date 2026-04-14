import json
import math
from typing import Any
import numpy as np

from src.file_handler import load_json

class ConstrainedDecoder:
    """Générateur de JSON contraint, simplifié pour être lisible."""

    def __init__(self, model: Any):
        self.model = model
        
        # 1. Charger le vocabulaire du modèle
        raw_vocab = load_json(self.model.get_path_to_vocab_file()) or {}

        # Identifiants (IDs) des tokens importants
        self.quote_id = self.model.encode('"').tolist()[0][0]
        self.comma_id = self.model.encode(",").tolist()[0][0]
        self.brace_id = self.model.encode("}").tolist()[0][0]

        # 2. Trier le vocabulaire pour isoler les nombres et le texte
        self.number_ids = set()
        self.string_ids = set()

        for text_bytes, tid in raw_vocab.items():
            # Nettoyer le format du texte renvoyé par le tokenizer
            text = text_bytes.decode("utf-8", "ignore") if isinstance(text_bytes, bytes) else str(text_bytes)
            text_propre = text.strip("Ġ▁ ▂▃▄▅▆▇█") # Retire les espaces spéciaux
            
            # Est-ce un nombre ? (Ne contient que des chiffres ou notation scientifique)
            if text_propre and all(c in "0123456789.-+eE" for c in text_propre):
                self.number_ids.add(tid)
                
            # C'est un texte standard (on rejette juste les guillemets ou symboles système)
            # REMARQUE : Utiliser 'if' et non pas 'elif' ! Un nombre peut parfaitement être inclus dans une "string"
            if '"' not in text and "\\" not in text and "<|" not in text:
                self.string_ids.add(tid)


    def decode(self, prompt_text: str, function_name: str, schema: dict[str, Any]) -> str:
        """Génère le JSON pas à pas pour la fonction."""
        
        # --- ETAPE 1: Initialiser la phrase avec le début du JSON ---
        texte_genere = f'{{"name": "{function_name}", "parameters": {{'
        phrase_initiale = f"Extract parameters into JSON.\nUser: {prompt_text}\nJSON: {texte_genere}"
        input_ids = self.model.encode(phrase_initiale).tolist()[0]
        
        cles_attendues = list(schema.keys())

        # --- ETAPE 2: Générer chaque paramètre un par un ---
        for i, nom_param in enumerate(cles_attendues):
            
            # Ajouter une virgule s'il y a des paramètres précédents
            if i > 0:
                texte_genere += ', '
                input_ids.extend(self.model.encode(", ").tolist()[0])
                
            # Vérifier le type attendu
            type_attendu = schema[nom_param].get("type", "string") if isinstance(schema[nom_param], dict) else getattr(schema[nom_param], "type", "string")
            
            # Stoppe le programme net si le type est farfelu (ex: "numbetdfsfr")
            types_autorises = ["string", "number", "integer", "float", "boolean", "bool"]
            if type_attendu not in types_autorises:
                raise ValueError(f"Type inconnu ou non supporté : '{type_attendu}'. Autorisés : {types_autorises}")

            est_nombre = type_attendu in ["number", "integer", "float"]
            est_boolean = type_attendu in ["boolean", "bool"]

            # Préparer la clé du tableau: "mon_parametre": 
            texte_cle = f'"{nom_param}": '
            if not (est_nombre or est_boolean):
                texte_cle += '"' # On ouvre les guillemets si ce n'est pas une primitive (nombre ou booléen) !
                
            texte_genere += texte_cle
            input_ids.extend(self.model.encode(texte_cle).tolist()[0])

            # --- ETAPE 3: Laisser le modèle deviner la valeur (avec triche/masque) ---
            valeur_courante = ""
            echecs_consecutifs = 0
            
            # On boucle pour générer les tokens (limite à 60 tokens par paramètre pour éviter l'infini)
            for _ in range(60):
                logits = self.model.get_logits_from_input_ids(input_ids)
                
                # Masque : on met toutes les probabilités à "-infini" (interdit)
                masque = [-math.inf] * len(logits)
                
                # On autorise seulement la liste qu'on veut
                if est_nombre:
                    autorises = self.number_ids | {self.comma_id, self.brace_id}
                elif est_boolean:
                    autorises = self.string_ids | {self.comma_id, self.brace_id}
                else:
                    autorises = self.string_ids | {self.quote_id}
                    
                # Restaurer la probabilité des tokens autorisés UNIQUEMENT S'ILS NE SONT PAS TROP MAUVAIS
                # Le modèle met aux alentours de -20 les probas des mots absurdes qu'il ne veut PAS dire.
                for tid in autorises:
                    if tid < len(logits):
                        masque[tid] = logits[tid]
                
                choix_id = int(np.argmax(masque))
                
                # Si le modèle ne veut ABSOLUMENT pas générer de token autorisé (parce que c'est absurde), 
                # il donne un score très bas (ex: < -10) même au "meilleur" choix de notre liste tronquée.
                if masque[choix_id] < -10:
                    echecs_consecutifs += 1
                else:
                    echecs_consecutifs = 0
                
                # Si le LLM est obligé de choisir des mots hors sujet 3 fois de suite, c'est que c'est mort!
                if echecs_consecutifs >= 3:
                    # On abandonne ce paramètre, on force une valeur par défaut !
                     valeur_defaut = "0" if est_nombre else ("false" if est_boolean else '')
                     if not valeur_courante:
                         texte_genere += valeur_defaut
                         input_ids.extend(self.model.encode(valeur_defaut).tolist()[0])
                     # S'il était au milieu d'un mot, on ferme juste
                     if not (est_nombre or est_boolean):
                         texte_genere += '"'
                         input_ids.append(self.quote_id)
                     break
                
                texte_choix = self.model.decode([choix_id])
                
                # Conditions pour S'ARRETER de générer le paramètre actuel:
                if not (est_nombre or est_boolean) and choix_id == self.quote_id:
                    texte_genere += '"' # Fermer les guillemets
                    input_ids.append(self.quote_id)
                    break
                    
                if (est_nombre or est_boolean) and choix_id in {self.comma_id, self.brace_id}:
                    if not valeur_courante: # S'il tente de fermer sans rien écrire, forcer default
                        valeur_defaut = "0" if est_nombre else "false"
                        texte_genere += valeur_defaut
                        input_ids.extend(self.model.encode(valeur_defaut).tolist()[0])
                    break
                
                # Sinon on ajoute le token à la valeur courante
                texte_genere += texte_choix
                valeur_courante += texte_choix
                input_ids.append(choix_id)

        # --- ETAPE 4: Clôturer proprement et vérifier le JSON ---
        texte_genere += '}}'
        
        try:
            json.loads(texte_genere)
            return texte_genere
        except json.JSONDecodeError:
            # Plan de secours ("Fallback") si le JSON est abîmé : on met des valeurs par défaut
            valeurs_defaut = {}
            for k, v in schema.items():
                if any(t in str(v) for t in ["number", "integer", "float"]):
                    valeurs_defaut[k] = 0
                elif any(t in str(v) for t in ["boolean", "bool"]):
                    valeurs_defaut[k] = False
                else:
                    valeurs_defaut[k] = ""
            return json.dumps({"name": function_name, "parameters": valeurs_defaut})
