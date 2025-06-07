from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
import pandas as pd
import torch
import gc
from sklearn.preprocessing import MinMaxScaler

from pykeen.predict import predict_target

from model_manager import *
from utils import *

# Configure logging
logger = logging.getLogger(__name__)

def _normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize scores in DataFrame using MinMaxScaler."""
    if df.empty:
        df["normalized_score"] = 0.0
        return df
    
    scaler = MinMaxScaler()
    df["normalized_score"] = scaler.fit_transform(df[["score"]])
    return df

def map_user_input_to_criteria(
    cooking_method: str,
    diet_types: List[str],
    meal_type: List[str],
    health_types: List[str],
    cuisine_region: str,
    ingredients: List[str],
    weights: Dict[str, float],
) -> List[Tuple[str, str, float]]:
    """
    Convert user input into criteria triples for prediction.
    
    Args:
        cooking_method: Cooking method preference
        diet_types: Diet type preferences
        meal_type: Meal type preferences
        health_types: Health attribute preferences
        cuisine_region: Cuisine region preference
        ingredients: Ingredient preferences
        weights: Importance weights for each criterion
        
    Returns:
        List of (tail_entity, relation, weight) triples
    """
    logger.info("Mapping user input to criteria")
    criteria = []
    default_weight = 1.0
    
    if cooking_method:
        cm = cooking_method.strip().lower()
        tail = f"cooking_method_{cm}"
        criteria.append((
            tail, 
            "usesCookingMethod", 
            weights.get("cooking_method", default_weight)
        ))

    for dt in diet_types:
        dt_clean = dt.strip()
        tail = f"diet_type_{dt_clean}"
        criteria.append((
            tail, 
            "hasDietType", 
            weights.get("diet_types", default_weight)
        ))

    for mt in meal_type:
        mt_clean = mt.strip()
        tail = f"meal_type_{mt_clean}"
        criteria.append((
            tail, 
            "isForMealType", 
            weights.get("meal_type", default_weight)
        ))

    for ht in health_types:
        ht_clean = ht.strip()
        relation = map_health_attribute(ht_clean)
        tail = f"health_attribute_{ht_clean}"
        criteria.append((
            tail, 
            relation, 
            weights.get("healthy_type", default_weight)
        ))

    if cuisine_region:
        cr = cuisine_region.strip()
        tail = f"cuisine_region_{cr}"
        criteria.append((
            tail, 
            "hasCuisineRegion", 
            weights.get("cuisine_region", default_weight)
        ))

    for ing in ingredients:
        ing_clean = ing.strip()
        tail = f"ingredient_{ing_clean}"
        criteria.append((
            tail, 
            "containsIngredient", 
            weights.get("ingredients", default_weight)
        ))

    logger.info(f"Created {len(criteria)} criteria from user input")
    return criteria

def get_matching_recipes(
    criteria: List[Tuple[str, str, float]], 
    MODEL_PATH: Path,
    TRIPLES_PATH: Path,
    top_k: int = 5, 
    flexible: bool = False
) -> List[str]:
    """
    Find recipes matching the given criteria.
    
    Args:
        criteria: List of (tail_entity, relation, weight) triples
        top_k: Number of recipes to return
        flexible: Whether to use flexible matching (OR) or strict matching (AND)
        
    Returns:
        List of matching recipe IDs
    """
    if not criteria:
        logger.warning("No criteria provided for recommendation")
        return []

    logger.info(f"Finding recipes matching {len(criteria)} criteria (flexible={flexible})")
    
    try:
        # Get model and triples factory from singleton
        model, triples_factory = model_manager.get_model_and_triples(MODEL_PATH, TRIPLES_PATH)
        
        all_preds = []
        for tail, relation, weight in criteria:
            try:
                # Use with torch.no_grad for memory efficiency
                with torch.no_grad():
                    preds = predict_target(
                        model=model, 
                        relation=relation, 
                        tail=tail, 
                        triples_factory=triples_factory
                    ).df
                
                preds = _normalize_scores(preds)
                preds["weighted_score"] = preds["normalized_score"] * weight
                preds = preds[["head_label", "weighted_score"]]
                all_preds.append(preds)
                
                # Clear unnecessary variables to free memory
                del preds
            except Exception as e:
                logger.error(f"Error predicting for {relation}, {tail}: {str(e)}")
                continue

        if not all_preds:
            logger.warning("No valid predictions obtained")
            return []
        
        # Process with careful memory management
        merged = all_preds[0].copy()
        for i, other in enumerate(all_preds[1:]):
            if flexible:
                # OR logic - keep recipes that match any criterion
                merged = merged.merge(
                    other, 
                    on="head_label", 
                    how="outer", 
                    suffixes=("", "_y")
                )
                merged["weighted_score"] = (
                    merged["weighted_score"].fillna(0) + 
                    merged["weighted_score_y"].fillna(0)
                )
                merged.drop(columns=["weighted_score_y"], inplace=True)
            else:
                # AND logic - only keep recipes that match all criteria
                merged = merged.merge(
                    other, 
                    on="head_label", 
                    how="inner", 
                    suffixes=("", "_y")
                )
                merged["weighted_score"] += merged["weighted_score_y"]
                merged.drop(columns=["weighted_score_y"], inplace=True)
            
            # Clean up to save memory
            all_preds[i+1] = None
            
        # Filter to only recipe nodes and sort by score
        merged = merged[merged["head_label"].str.startswith("recipe_")]
        merged.sort_values(by="weighted_score", ascending=False, inplace=True)
        merged = merged.head(top_k)

        # Extract recipe IDs from node labels
        def parse_recipe_id(node_str: str) -> str:
            return node_str.split("recipe_", 1)[1]

        ids = merged["head_label"].apply(parse_recipe_id).to_list()
        logger.info(f"Found {len(ids)} matching recipes")
        
        # Clean up for good measure
        del merged, all_preds
        
        return ids
    
    finally:
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def fetch_recipe_info(recipe_id: str, RECIPE_PATH_CSV: Path) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed information for a specific recipe.
    
    Args:
        recipe_id: Recipe ID
        
    Returns:
        Recipe information or None if not found
    """
    try:
        rid_int = int(recipe_id)
        logger.info(f"Fetching info for recipe ID: {rid_int}")
    except ValueError:
        logger.error(f"Invalid recipe ID: {recipe_id}")
        return None

    recipes_df = pd.read_csv(RECIPE_PATH_CSV)
    row = recipes_df[recipes_df["RecipeId"] == rid_int]
    
    if row.empty:
        logger.warning(f"Recipe not found: {rid_int}")
        return None
    
    result = row.iloc[0].to_dict()
    logger.info(f"Found recipe: {result.get('Name', 'Unknown')}")
    return result