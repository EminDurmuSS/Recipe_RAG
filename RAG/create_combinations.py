
import pandas as pd
import json
import itertools
import random
from typing import List

def split_ingredients(cell: str) -> List[str]:
    """Split semicolon‑delimited ingredients & strip whitespace. Return [] for NaNs."""
    if pd.isna(cell):
        return []
    return [item.strip() for item in str(cell).split(';') if item.strip()]

def generate_random_combinations(k, max_combinations):
    """
    Generate random combinations of criteria and values for experimental design.

    Parameters:
        criteria (dict): Dictionary of criteria and values.
        k (int): Number of keys to combine.
        max_combinations (int): Max number of combinations to return.

    Returns:
        List of dictionaries with randomly selected key-value combinations.
    """

    criteria = {
    "cooking_method":["oven", "pot", "pan", "no cook", "barbecue", "air fryer", "microwave"],
    "diet_types":["Vegetarian", "Vegan", "Paleo", "Standard"],
    "meal_type":["breakfast", "lunch", "dinner", "snack", "dessert", "starter", "brunch", "drink"],
    "health_types":["Low Protein", "Medium Protein", "High Protein",
                    "Low Carb", "Medium Carb", "High Carb",
                    "Low Fat", "Medium Fat", "High Fat",
                    "Low Calorie", "Medium Calorie", "High Calorie"],
    "cuisine_region":["North America", "Global", "Mediterranean Europe", "Northern/Western Europe", "Latin America",
                     "East Asia", "South Asia", "Southeast Asia", "Middle East & Anatolia",
                     "Oceania", "Eastern Europe & Eurasia", "Caribbean", "Sub-Saharan Africa"] 
    }

    keys_to_wrap = ['diet_types', 'meal_type', 'health_types']

    keys = list(criteria.keys())
    key_combinations = list(itertools.combinations(keys, k))
    random.shuffle(key_combinations)  # Shuffle to vary key sets

    selected_combinations = []
    tried = 0

    for key_combo in key_combinations:
        value_lists = [criteria[key] for key in key_combo]
        
        # Instead of generating ALL products, sample just a few (or one)
        value_sample = [random.choice(vs) for vs in value_lists]
        temp = dict(zip(key_combo, value_sample))
        wrapped = {k: [v] if k in keys_to_wrap else v
                   for k, v in temp.items()}
        selected_combinations.append(wrapped)

        tried += 1
        if len(selected_combinations) >= max_combinations:
            break

    return selected_combinations

def generate_random_ingredient_combinations(RECIPE_PATH_CSV, k, max_combinations):
    """
    Efficiently sample random unique combinations of k ingredients.

    Parameters:
        ingredients (list): List of ingredients.
        k (int): Number of ingredients per combination.
        max_combinations (int): Number of unique combinations to return.

    Returns:
        List of unique combinations (tuples of ingredients).
    """
    df = pd.read_csv(RECIPE_PATH_CSV)
    df['ingredient_list'] = df['BestUsdaIngredientName'].apply(split_ingredients)
    ingredients = [ing for lst in df['ingredient_list'] for ing in lst]

    if k > len(ingredients):
        raise ValueError("k cannot be greater than the number of ingredients.")

    seen = set()
    combinations = []

    attempts = 0
    max_attempts = max_combinations * 10  # to avoid infinite loops

    while len(combinations) < max_combinations and attempts < max_attempts:
        combo = tuple(sorted(random.sample(ingredients, k)))
        if combo not in seen:
            seen.add(combo)
            combinations.append({'ingredients':list(combo)})
        attempts += 1

    return combinations

def save_json(json_path, json_text):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_text, f, indent=2, ensure_ascii=False)

def create_combinations(RECIPE_PATH_CSV, JSON_PATH, max_combinations):
    combos_without_ingredients_1 = generate_random_combinations(1, max_combinations)
    combos_without_ingredients_2 = generate_random_combinations(2, max_combinations)
    combos_without_ingredients_3 = generate_random_combinations(3, max_combinations)
    combos_without_ingredients_combined = combos_without_ingredients_1 + \
                                          combos_without_ingredients_2 + \
                                          combos_without_ingredients_3
    save_json(JSON_PATH + '_without_ingredients.json', combos_without_ingredients_combined)

    combos_with_ingredients_1 = generate_random_ingredient_combinations(RECIPE_PATH_CSV, 1, max_combinations)
    combos_with_ingredients_2 = generate_random_ingredient_combinations(RECIPE_PATH_CSV, 2, max_combinations)
    combos_with_ingredients_3 = generate_random_ingredient_combinations(RECIPE_PATH_CSV, 3, max_combinations)
    combos_with_ingredients_combined = combos_with_ingredients_1 + \
                                       combos_with_ingredients_2 + \
                                       combos_with_ingredients_3
    save_json(JSON_PATH + '_with_ingredients.json', combos_with_ingredients_combined)

    return combos_without_ingredients_combined, combos_with_ingredients_combined