from pydantic import BaseModel, Field
from typing import Dict, Optional


class PropertySchema(BaseModel):
    """Schéma d'une propriété dans les paramètres."""
    type: str


class FunctionReturns(BaseModel):
    """Schéma du retour d'une fonction."""
    type: str


class FunctionDefinition(BaseModel):
    """Définition complète d'une fonction.

    Provient de functions_definitions.json.
    """
    name: str
    description: str
    parameters: Dict[str, PropertySchema] = Field(default_factory=dict)
    returns: Optional[FunctionReturns] = None


class PromptInput(BaseModel):
    """Entree utilisateur"""
    prompt: str
