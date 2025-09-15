import ast
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from pykeen.models import Model
from pykeen.triples import TriplesFactory

logger = logging.getLogger(__name__)

def tuple_to_canonical(s: str) -> str:
    """
    Converts a string tuple representation into canonical format.
    Example: "('meal_type', 'dinner')" -> "meal_type_dinner"
    """
    try:
        t = ast.literal_eval(s)
        if not isinstance(t, tuple) or len(t) != 2:
            logger.warning(f"Expected tuple of length 2, got: {s}")
            return s
        return f"{t[0]}_{t[1]}"
    except Exception as e:
        logger.error(f"Error converting tuple '{s}': {str(e)}")
        return s

def load_kge_model(MODEL_PATH: Path) -> Model:
    """Load the knowledge graph embedding model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    try:
        logger.info(f"Loading KGE model from {MODEL_PATH}")
        return torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def get_triples_factory(TRIPLES_PATH: Path) -> TriplesFactory:
    """Create a TriplesFactory from the triples CSV file."""
    if not TRIPLES_PATH.exists():
        raise FileNotFoundError(f"Triples file not found: {TRIPLES_PATH}")
    
    try:
        logger.info(f"Loading triples from {TRIPLES_PATH}")
        df = pd.read_csv(TRIPLES_PATH)
        
        # Convert each Head, Relation, Tail string into a standardized format
        triples = []
        for h, r, t in df[["Head", "Relation", "Tail"]].values:
            ch = tuple_to_canonical(h)
            cr = r.strip()
            ct = tuple_to_canonical(t)
            triples.append((ch, cr, ct))
        
        logger.info(f"Loaded {len(triples)} triples")
        return TriplesFactory.from_labeled_triples(
            triples=np.array(triples, dtype=str), 
            create_inverse_triples=False
        )
    except Exception as e:
        logger.error(f"Failed to create triples factory: {str(e)}")
        raise

def map_health_attribute(element: str) -> str:
    """Map health attribute string to a relation name."""
    e = element.lower()
    if "protein" in e:
        return "HasProteinLevel"
    elif "carb" in e:
        return "HasCarbLevel"
    elif "fat" in e and "saturated" not in e:
        return "HasFatLevel"
    elif "saturated_fat" in e:
        return "HasSaturatedFatLevel"
    elif "calorie" in e:
        return "HasCalorieLevel"
    elif "sodium" in e:
        return "HasSodiumLevel"
    elif "sugar" in e:
        return "HasSugarLevel"
    elif "fiber" in e:
        return "HasFiberLevel"
    elif "cholesterol" in e:
        return "HasCholesterolLevel"
    else:
        return "HasHealthAttribute"
    
def format_recipe_example(recipe: dict) -> str:
    """
    Format a recipe dictionary to display certain keys as an example for the LLM prompt.
    """
    keys = [
        "Name", "Description", "RecipeCategory", "Keywords", "RecipeInstructions",
        "cook_time", "Healthy_Type", "Diet_Types", "meal_type", "ScrapedIngredients",
        "CuisineRegion", "Cooking_Method", "servings_bin"
    ]
    output_lines = []
    for key in keys:
        value = recipe.get(key, "Not provided")
        output_lines.append(f"{key}: {value}")
    return "\n".join(output_lines)

def format_user_criteria(criteria: dict) -> str:
    lines = []
    lines.append("User Criteria:")

    cooking_method = criteria.get("cooking_method", [])
    if isinstance(cooking_method, list):
        lines.append(f"  - Cooking Method(s): {', '.join(cooking_method) or 'Any'}")
    else:
        lines.append(f"  - Cooking Method(s): {cooking_method or 'Any'}")

    lines.append(f"  - Cuisine Region: {criteria.get('cuisine_region', 'Any')}")

    diet_types = criteria.get("diet_types", [])
    if isinstance(diet_types, list):
        lines.append(f"  - Diet Types: {', '.join(diet_types) or 'Any'}")
    else:
        lines.append(f"  - Diet Types: {diet_types or 'Any'}")

    meal_type = criteria.get("meal_type", [])
    if isinstance(meal_type, list):
        lines.append(f"  - Meal Types: {', '.join(meal_type) or 'Any'}")
    else:
        lines.append(f"  - Meal Types: {meal_type or 'Any'}")

    # Include cook time and servings for the LLM prompt (but not for KG matching)
    lines.append(f"  - Cooking Time: {criteria.get('cook_time', 'Any time')}")
    lines.append(f"  - Servings: {criteria.get('servings_bin', 'Any servings')}")

    ingredients = criteria.get("ingredients", [])
    if isinstance(ingredients, list):
        lines.append(f"  - Ingredients: {', '.join(ingredients) or 'None specified'}")
    else:
        lines.append(f"  - Ingredients: {ingredients or 'None specified'}")

    health_types = criteria.get("health_types", [])
    if isinstance(health_types, list):
        lines.append(f"  - Health Types: {', '.join(health_types) or 'Any'}")
    else:
        lines.append(f"  - Health Types: {health_types or 'Any'}")

    return "\n".join(lines)