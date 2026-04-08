# Résumé du Projet : Call Me Maybe (Function Calling)

## But du Projet

Ce projet a pour but de vous apprendre à implémenter un système de **Function Calling** (Appel de fonction) en utilisant un "petit" modèle de langage (Large Language Model - Qwen3-0.6B).

Habituellement, quand on demande à un LLM de lire une commande (ex: "Combien font 40 + 2 ?") et de renvoyer un format strict (comme un JSON avec le nom de la fonction et ses arguments), les petits modèles se trompent très souvent sur la syntaxe.

Votre mission est de corriger ce problème en utilisant une technique appelée **Décodage Contraint (Constrained Decoding)**. Au lieu de laisser le modèle deviner le JSON, vous allez intervenir lors de la génération de chaque mot/morceau de mot (token). En modifiant les probabilités (logits) retournées par le modèle, vous allez littéralement "forcer" le modèle à n'écrire que du JSON 100% valide qui respecte la structure (le schéma) que vous attendez.

## Règles Importantes

- **Librairies autorisées :** `numpy`, `pydantic`, `json`.
- **Interdit :** Utiliser des librairies de Machine Learning lourdes comme `pytorch`, `transformers`, `huggingface`, etc.
- Vous devez utiliser le package interne `llm_sdk` fourni avec le projet.
- La validation des données doit obligatoirement passer par `pydantic`.
- Tolérance zéro pour les plantages (crash) : votre code doit gérer toutes les erreurs proprement.

---

## Todo-Liste

- [x] **Étape 1 : Environnement** - Initialiser le projet avec `uv` (installer numpy, pydantic). _(Fait !)_
- [x] **Étape 2 : Makefile** - Créer les règles `install`, `run`, `debug`, `clean`, et `lint` (avec un typage strict). _(Fait !)_
- [ ] **Étape 3 : Interface en Ligne de Commande (CLI)** - Dans `src/__main__.py`, utiliser `argparse` pour lire les arguments `--functions_definition`, `--input`, `--output`. Gérer les erreurs de fichiers.
- [ ] **Étape 4 : Modèles de données (Pydantic)** - Dans `src/data_models.py`, créer des classes Pydantic pour représenter la structure attendue des fonctions (nom, paramètres, types).
- [ ] **Étape 5 : Cartographie du Vocabulaire (Tokens)** - Écrire une logique pour lire le fichier contenant le vocabulaire du modèle, afin de savoir à quel texte correspond chaque "Token ID".
- [ ] **Étape 6 : Cœur du projet (Décodage Contraint)** - Implémenter la logique qui intercepte les logits (probabilités) du modèle (`Small_LLM_Model`), et met à `-inf` (probabilité nulle) tous les tokens qui casseraient le format JSON ou le schéma Pydantic de la fonction.
- [ ] **Étape 7 : Boucle principale** - Lire les questions depuis le fichier d'entrée, faire générer le JSON par votre système, le valider, puis l'écrire dans le fichier de sortie.
- [ ] **Étape 8 : Finitions & Linting** - S'assurer que les commandes `make lint` passent sans erreur (mypy, flake8).
