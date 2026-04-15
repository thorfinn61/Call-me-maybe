from typing import Dict
from pydantic import BaseModel, ConfigDict


class PropertySchema(BaseModel):
    """Schéma d'une propriété dans les paramètres."""
    model_config = ConfigDict(extra="forbid")
    type: str


class FunctionReturns(BaseModel):
    """Schéma du retour d'une fonction."""
    model_config = ConfigDict(extra="forbid")
    type: str


class FunctionDefinition(BaseModel):
    """Définition complète d'une fonction.

    Provient de functions_definitions.json.
    """
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: Dict[str, PropertySchema]
    returns: FunctionReturns


class PromptInput(BaseModel):
    """Entree utilisateur"""
    model_config = ConfigDict(extra="forbid")
    prompt: str
