NAME = src/__main__.py

VENV = .venv
UV = uv

GREEN = \033[0;32m
RESET = \033[0m

all: run

install:
	@echo "$(GREEN)Initialisation de l'environnement avec uv...$(RESET)"
	$(UV) venv
	$(UV) pip install pydantic numpy flake8 mypy

run:
	@echo "$(GREEN)Lancement du programme...$(RESET)"
	$(UV) run python $(NAME)

debug:
	@echo "$(GREEN)Mode debug (PDB)...$(RESET)"
	$(UV) run python -m pdb $(NAME)

clean:
	@echo "$(GREEN)Nettoyage complet...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf $(VENV)
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type f -name "*.pyc" -delete

lint:
	@echo "$(GREEN)Vérification flake8...$(RESET)"
	$(UV) run flake8 src/
	@echo "$(GREEN)Vérification mypy (Mode Strict)...$(RESET)"
	$(UV) run mypy --strict --ignore-missing-imports src/

.PHONY: all install run debug clean lint