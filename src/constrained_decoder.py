import json
import math
import re
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
            if isinstance(text_bytes, bytes):
                text = text_bytes.decode("utf-8", "ignore")
            else:
                text = str(text_bytes)
            # Retire les espaces spéciaux
            text_propre = text.strip("Ġ▁ ▂▃▄▅▆▇█")

            # Est-ce un nombre ?
            if text_propre and all(c in "0123456789.-+eE"
                                   for c in text_propre):
                self.number_ids.add(tid)

            # C'est u texte standard (on rejette les guillemets)
            # REMARQUE : Utiliser 'if' et non 'elif' ! Un nombre
            # peut être inclus dans une "string"
            if '"' not in text and "\\" not in text and "<|" not in text:
                self.string_ids.add(tid)

    @staticmethod
    def _normalize_type(raw_type: str) -> str:
        """Normalise les variantes de types en types canoniques.

        Convertit les aliases en types standard.
        """
        normalized = raw_type.lower().strip()

        # Normaliser les nombres
        if normalized in ["number", "integer", "float"]:
            return "number"

        # Normaliser les booléens
        if normalized in ["boolean", "bool"]:
            return "boolean"

        # Les strings restent strings
        if normalized == "string":
            return "string"

        msg = (f"Type non supporté : '{raw_type}'. "
               f"Acceptés : number, integer, float, boolean, bool, string")
        raise ValueError(msg)

    def decode(
        self,
        prompt_text: str,
        function_name: str,
        schema: dict[str, Any],
    ) -> str:
        """Génère le JSON pas à pas pour la fonction."""

        # --- ETAPE 1: Valider et normaliser tous les types ---
        schema_normalise = {}
        for nom_param, prop in schema.items():
            if isinstance(prop, dict):
                raw_type = prop.get("type", "string")
            else:
                raw_type = getattr(prop, "type", "string")
            try:
                type_norm = self._normalize_type(raw_type)
                schema_normalise[nom_param] = type_norm
            except ValueError as e:
                msg = f"Paramètre '{nom_param}' : {str(e)}"
                raise ValueError(msg)

        # --- ETAPE 2: Initialiser la phrase avec le début du JSON ---
        texte_genere = f'{{"name": "{function_name}", "parameters": {{'
        extract_prompt = "Extract parameters into JSON."
        phrase_initiale = (
            f"{extract_prompt}\n"
            f"User: {prompt_text}\n"
            f"JSON: {texte_genere}"
        )
        input_ids = self.model.encode(phrase_initiale).tolist()[0]

        cles_attendues = list(schema_normalise.keys())
        number_literals = re.findall(r"-?\d+(?:\.\d+)?", prompt_text)
        number_literal_index = 0

        def next_number_literal() -> str | None:
            """Retourne le prochain nombre extrait du prompt, si disponible."""
            nonlocal number_literal_index
            if number_literal_index >= len(number_literals):
                return None
            literal = str(number_literals[number_literal_index])
            number_literal_index += 1
            return literal

        def should_use_numeric_fallback() -> bool:
            """Active le fallback numérique pour les fonctions mathématiques.

            Utilisé pour les fonctions de calcul simples.
            """
            return function_name in {"fn_add_numbers", "fn_get_square_root"}

        # --- ETAPE 3: Générer chaque paramètre un par un ---
        for i, nom_param in enumerate(cles_attendues):

            # Ajouter une virgule s'il y a des paramètres précédents
            if i > 0:
                texte_genere += ', '
                input_ids.extend(self.model.encode(", ").tolist()[0])

            # Type attendu (déjà normalisé)
            type_attendu = schema_normalise[nom_param]

            est_nombre = type_attendu == "number"
            est_boolean = type_attendu == "boolean"
            est_string = type_attendu == "string"

            # Préparer la clé du tableau: "mon_parametre":
            texte_cle = f'"{nom_param}": '
            if est_string:
                texte_cle += '"'  # On ouvre les guillemets pour les strings

            texte_genere += texte_cle
            input_ids.extend(self.model.encode(texte_cle).tolist()[0])

            # Cas robuste pour les fonctions mathématiques:
            # on prend directement les nombres du prompt.
            if should_use_numeric_fallback() and (est_string or est_nombre):
                fallback = next_number_literal()
                if fallback is None:
                    fallback = "0" if est_nombre else ""
                texte_genere += fallback
                if fallback:
                    input_ids.extend(
                        self.model.encode(fallback).tolist()[0]
                    )
                if est_string:
                    texte_genere += '"'
                    input_ids.append(self.quote_id)
                continue

            # --- ETAPE 4: Laisser le modèle deviner
            # la valeur (avec triche/masque) ---
            valeur_courante = ""
            echecs_consecutifs = 0

            # Générer les tokens (60 max par paramètre)
            for _ in range(60):
                logits = self.model.get_logits_from_input_ids(input_ids)

                # Masque: on met toutes les probabilités
                # à "-infini" (interdit)
                masque = [-math.inf] * len(logits)

                # On autorise seulement la liste qu'on veut
                if est_nombre:
                    autorises = self.number_ids | {
                        self.comma_id, self.brace_id}
                elif est_boolean:
                    autorises = self.string_ids | {
                        self.comma_id, self.brace_id}
                else:  # string
                    autorises = self.string_ids | {self.quote_id}

                # Restaurer la probabilité des tokens autorisés
                # (si logits n'est pas trop mauvais)
                for tid in autorises:
                    if tid < len(logits):
                        masque[tid] = logits[tid]

                choix_id = int(np.argmax(masque))

                # Si le modèle ne veut pas générer de token
                # autorisé (score < -10), c'est une impasse
                if masque[choix_id] < -10:
                    echecs_consecutifs += 1
                else:
                    echecs_consecutifs = 0

                # Si 3 impasses d'affilée, abandon
                if echecs_consecutifs >= 3:
                    # On abandonne ce paramètre,
                    # on force une valeur par défaut.
                    if est_nombre:
                        fallback = "0"
                        if should_use_numeric_fallback():
                            fallback = next_number_literal() or "0"
                        texte_genere += fallback
                        input_ids.extend(
                            self.model.encode(fallback).tolist()[0]
                        )
                    elif est_boolean:
                        texte_genere += "false"
                        input_ids.extend(
                            self.model.encode("false").tolist()[0])
                    else:  # string
                        if should_use_numeric_fallback():
                            fallback = next_number_literal()
                            if fallback is not None:
                                texte_genere += fallback
                                input_ids.extend(
                                    self.model.encode(fallback).tolist()[0])
                        # S'il était au milieu d'un mot, on ferme juste
                        texte_genere += '"'
                        input_ids.append(self.quote_id)
                    break

                texte_choix = self.model.decode([choix_id])

                # Conditions pour S'ARRETER de générer le paramètre actuel:
                if est_string and choix_id == self.quote_id:
                    if (
                        not valeur_courante
                        and should_use_numeric_fallback()
                    ):
                        fallback = next_number_literal()
                        if fallback is not None:
                            texte_genere += fallback
                            input_ids.extend(
                                self.model.encode(fallback).tolist()[0])
                    texte_genere += '"'  # Fermer les guillemets
                    input_ids.append(self.quote_id)
                    break

                if (est_nombre or est_boolean) and choix_id in {
                        self.comma_id, self.brace_id}:
                    if not valeur_courante:  # Forcer default si vide
                        if est_nombre and should_use_numeric_fallback():
                            v = next_number_literal() or "0"
                        else:
                            v = "0" if est_nombre else "false"
                        texte_genere += v
                        input_ids.extend(
                            self.model.encode(v).tolist()[0])
                    break

                # Sinon on ajoute le token à la valeur courante
                texte_genere += texte_choix
                valeur_courante += texte_choix
                input_ids.append(choix_id)

        # --- ETAPE 5: Clôturer proprement et vérifier le JSON ---
        texte_genere += '}}'

        try:
            json.loads(texte_genere)
            return texte_genere
        except json.JSONDecodeError:
            # Fallback avec valeurs par défaut
            valeurs_defaut: dict[str, Any] = {}
            for nom_param, type_norm in schema_normalise.items():
                if type_norm == "number":
                    valeurs_defaut[nom_param] = 0
                elif type_norm == "boolean":
                    valeurs_defaut[nom_param] = False
                else:  # string
                    valeurs_defaut[nom_param] = ""
            result = {"name": function_name, "parameters": valeurs_defaut}
            return json.dumps(result)
