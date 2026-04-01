NAME = call_me_maybe.py

PYTHON = python3
PIP = pip

GREEN = \033[0;32m
RESET = \033[0m

all: run

install:
	@echo "$(GREEN)Installation des dépendances...$(RESET)"
	$(PIP) install flake8 mypy pytest

run:
	$(PYTHON) $(NAME) 

debug:
	$(PYTHON) -m pdb $(NAME)

clean:
	@echo "$(GREEN)Nettoyage complet...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type f -name "*.pyc" -delete

lint:
	@echo "$(GREEN)Vérification flake8...$(RESET)"
	flake8 .
	@echo "$(GREEN)Vérification mypy...$(RESET)"
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		 --disallow-untyped-defs --check-untyped-defs .

.PHONY: all install run debug clean lint build