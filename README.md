*This project has been created as part of the 42 curriculum by elsahin.*

# Call me maybe

## Description
Ce projet implemente un pipeline de function calling base sur un petit modele de langage, avec une generation JSON controlee par constrained decoding.

Objectif principal:
- lire un prompt utilisateur,
- selectionner la fonction la plus pertinente,
- generer un JSON valide avec les parametres,
- garantir la robustesse face aux erreurs de schema et d'entree.

Le coeur du projet est dans [src/constrained_decoder.py](src/constrained_decoder.py), avec orchestration CLI dans [src/__main__.py](src/__main__.py).

## Instructions
### Prerequis
- Python 3.13+
- `uv` installe

### Installation
```bash
make install
```

### Execution
```bash
make run
```

### Execution avec options
```bash
uv run python -m src \
  --functions_definition src/data/input/functions_definition.json \
  --input src/data/input/function_calling_tests.json \
  --output src/data/output/function_calling_results.json
```

### Debug
```bash
make debug
```

### Qualite de code
```bash
make lint
```

### Nettoyage
```bash
make clean
```

## Algorithm explanation
Approche de constrained decoding utilisee:

1. Initialisation
- chargement du vocabulaire token du modele,
- construction de 2 ensembles de tokens: nombres (`number_ids`) et texte (`string_ids`),
- memorisation des tokens de structure JSON (`"`, `,`, `}`).

2. Validation du schema d'entree
- chaque type de parametre est normalise (`number`, `integer`, `float`, `boolean`, `bool`, `string`),
- tout type non supporte declenche une erreur immediate (`ValueError`) avec contexte du parametre.

3. Generation parametre par parametre
- pour chaque cle du schema, le decodeur force la forme JSON attendue,
- a chaque step, un masque de logits est applique:
  - type `number`: seuls tokens numeriques + tokens de fermeture sont autorises,
  - type `boolean`: tokens texte + tokens de fermeture,
  - type `string`: tokens texte + guillemet de fermeture.

4. Strategie d'arret et fallback
- un compteur d'impasse evite les boucles longues,
- apres plusieurs logits invalides consecutifs, fallback sur valeur par defaut,
- verification finale via `json.loads`, puis fallback global si JSON invalide.

5. Robustesse metier supplementaire
- cas regex: post-traitement de [src/__main__.py](src/__main__.py) pour corriger les parametres de substitution,
- cas prompts mathematiques: extraction des litteraux numeriques dans le prompt pour stabiliser la sortie.

## Design decisions
Choix principaux:

- Pydantic pour valider les fichiers d'entree et garantir un schema strict.
- `extra="forbid"` dans [src/models.py](src/models.py) pour refuser les fautes de frappe (`ss`, `paramters`, etc.).
- Arret propre sur erreur de validation dans [src/__main__.py](src/__main__.py), avec message lisible plutot qu'un crash brut.
- Separation nette des responsabilites:
  - selection de fonction dans [src/function_selector.py](src/function_selector.py),
  - decoding contraint dans [src/constrained_decoder.py](src/constrained_decoder.py),
  - I/O JSON dans [src/file_handler.py](src/file_handler.py).

## Performance analysis
### Accuracy
- bonne adherence au schema JSON pour les cas testes,
- meilleure robustesse quand les types sont valides,
- comportement protege par validation stricte quand le schema est invalide.

### Speed
- affichage du temps total d'execution dans [src/__main__.py](src/__main__.py),
- generation token-by-token plus lente qu'une generation libre, mais necessaire pour la contrainte forte.

### Reliability
- validation des inputs avant inference,
- gestion explicite des erreurs de parsing/validation,
- fallback JSON pour limiter les crashes.

## Challenges faced
Problemes rencontres et resolutions:

- Boucles longues / generation incoherente
  - resolution: limites de generation et strategie de fallback.

- Erreurs silencieuses de schema (`name` renomme, `parameters` mal ecrit)
  - resolution: modeles Pydantic stricts avec `extra="forbid"` et champs obligatoires.

- Variabilite du petit modele sur certains prompts
  - resolution: post-traitement cible (regex) et extraction numerique pour les cas mathematiques.

## Testing strategy
Strategie de validation:

- tests fonctionnels via le dataset de prompts dans [src/data/input/function_calling_tests.json](src/data/input/function_calling_tests.json),
- verification des sorties dans [src/data/output/function_calling_results.json](src/data/output/function_calling_results.json),
- validation statique avec `flake8` et `mypy --strict` (`make lint`),
- tests de robustesse manuels en injectant des schemas invalides (ex: `ss` au lieu de `name`, `paramters` au lieu de `parameters`).

## Example usage
### Cas nominal
```bash
make run
```

Exemple de log:
```text
🚀 Lancement du Constrained Decoding Runner...
[1/11] ✅ Succès (...)
...
🎉 Terminé ! Les résultats sont dans src/data/output/function_calling_results.json
⏱️ Temps total : 153.12s
```

### Cas schema invalide
Si `functions_definition.json` contient une faute de cle (ex: `ss` ou `paramters`), le programme affiche une erreur de validation claire puis s'arrete proprement.

## Resources
References classiques:
- JSON Specification: https://www.rfc-editor.org/rfc/rfc8259
- Pydantic documentation: https://docs.pydantic.dev/
- Python `argparse`: https://docs.python.org/3/library/argparse.html
- Python `json`: https://docs.python.org/3/library/json.html
- Constrained decoding (overview):
  - https://huggingface.co/blog/constrained-beam-search
  - https://arxiv.org/abs/2307.09702 (structured generation context)

Utilisation de l'IA dans ce projet:
- assistance pour refactoring du decodeur et clarification de la logique,
- aide a la mise en place de la gestion d'erreurs et des messages utilisateur,
- aide pour le constrained decoding
- aide ponctuelle sur linting/typing et reformulation de documentation.

L'implementation finale, les choix techniques et les validations ont ete verifies et ajustes dans le code du repository.
