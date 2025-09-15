import pandas as pd
import json
import itertools
import random
from collections import Counter
from typing import List

def split_ingredients(cell: str) -> List[str]:
    """Split semicolon‑delimited ingredients & strip whitespace. Return [] for NaNs."""
    if pd.isna(cell):
        return []
    return [item.strip() for item in str(cell).split(';') if item.strip()]

def get_frequent_ingredients():
    df = pd.read_csv("../data/dataFullLargerRegionAndCountryWithServingsBin.csv")
    df['ingredient_list'] = df['BestUsdaIngredientName'].apply(split_ingredients)
    ingredients = [ingr for lst in df['ingredient_list'] for ingr in lst]
    ingr_counter = Counter(ingredients)
    df_ingr = (
        pd.DataFrame(ingr_counter.items(), columns=['ingredient', 'count'])
        .sort_values('count', ascending=False)
        .reset_index(drop=True)
    )
    df_ingr['bin'] = pd.qcut(df_ingr['count'], q=3, labels=['rare', 'medium', 'frequent'])
    return df_ingr[df_ingr['bin'] == 'frequent']['ingredient'].tolist()

CRITERIA = {
    "cooking_method": ["oven", "pot", "pan", "no cook", "barbecue", "air fryer", "microwave"],
    "diet_types": ["Vegetarian", "Vegan", "Paleo", "Standard"],
    "meal_type": ["breakfast", "lunch", "dinner", "snack", "dessert", "starter", "brunch", "drink"],
    "health_types": [
        "Low Protein", "Medium Protein", "High Protein",
        "Low Carb", "Medium Carb", "High Carb",
        "Low Fat", "Medium Fat", "High Fat",
        "Low Calorie", "Medium Calorie", "High Calorie"
    ],
    "cuisine_region": [
        "North America", "Global", "Mediterranean Europe", "Northern/Western Europe", "Latin America",
        "East Asia", "South Asia", "Southeast Asia", "Middle East & Anatolia",
        "Oceania", "Eastern Europe & Eurasia", "Caribbean", "Sub-Saharan Africa"
    ]
}

WRAP_KEYS = {"diet_types", "meal_type", "health_types"}

INGREDIENTS = get_frequent_ingredients()

def generate_random_combination(k):
    """
    Generate random combination of criteria.
    """
    keys = list(CRITERIA.keys())
    key_combinations = list(itertools.combinations(keys, k))
    random.shuffle(key_combinations)
    combination = {
        k: [random.choice(CRITERIA[k])] if k in WRAP_KEYS else random.choice(CRITERIA[k])
        for k in key_combinations[0]
    }
    return combination

def generate_random_ingredient_combination(k):
    """
    Efficiently sample unique combinations of k ingredients.
    """
    combo = tuple(sorted(random.sample(INGREDIENTS, k)))
    combinations = {'ingredients': list(combo)}
    return combinations

def save_json(json_path, json_text):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_text, f, indent=2, ensure_ascii=False)

def create_combination():
    """
    Create random combinations of either user-defined criteria or ingredient sets.
    """
    k = random.randint(2, 5)  
    f1 = generate_random_combination(k)
    f2 = generate_random_ingredient_combination(k)
    return random.choice([f1, f2])