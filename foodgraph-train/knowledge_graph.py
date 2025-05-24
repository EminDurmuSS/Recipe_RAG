"""
This module builds a directed knowledge graph (KG) from recipe data for a food recommendation system.
It reads a CSV file containing recipe details, extracts only the selected columns, and constructs the KG.
The selected columns are:
    - RecipeId (unique recipe identifier)
    - Cooking_Method
    - Diet_Types
    - meal_type
    - Healthy_Type (processed with custom mapping)
    - CuisineRegion (represents the country/region of the dish)
    - BestUsdaIngredientName (ingredient names, semicolon-delimited)

Each recipe is represented as a node with a tuple identifier ("recipe", RecipeId), and each attribute is
represented as a node with a tuple identifier ("<type>", value). Directed edges are added from the recipe
node to the attribute node with a specified relation label. The resulting triples (Head, Relation, Tail)
are stored for later use in embedding-based recommendations.

Author: [Emin Durmuş]
Date: [1.02.2025]
"""

import numpy as np
import networkx as nx
import pickle
import pandas as pd
from typing import Dict, Any, Tuple, List


UNKNOWN_PLACEHOLDER = "unknown"

# ---------------------------
# Helper Functions
# ---------------------------


def map_health_attribute(element: str) -> str:
    """
    Map a healthy attribute string to a specific relation label.

    Parameters:
        element (str): The healthy attribute value.

    Returns:
        str: A relation label (e.g., 'HasProteinLevel').
    """
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


def split_and_clean(value: str, delimiter: str) -> List[str]:
    """
    Splits a string by the specified delimiter and trims whitespace from each element.

    Parameters:
        value (str): The string to split.
        delimiter (str): The delimiter used for splitting.

    Returns:
        List[str]: A list of cleaned substrings.
    """
    return [v.strip() for v in value.split(delimiter) if v.strip()]


def load_recipes_from_dataframe(df: pd.DataFrame) -> Dict[Any, Dict[str, Any]]:
    """
    Load and filter recipes from a DataFrame into a dictionary keyed by RecipeId.

    Only the following columns are retained:
        - RecipeId, Cooking_Method, Diet_Types, meal_type,
          Healthy_Type, CuisineRegion, BestUsdaIngredientName

    Parameters:
        df (pd.DataFrame): DataFrame loaded from CSV.

    Returns:
        Dict[Any, Dict[str, Any]]: Dictionary where keys are RecipeId and values are dictionaries of attributes.
    """
    columns_to_keep = [
        "RecipeId",
        "Cooking_Method",
        "Diet_Types",
        "meal_type",
        "Healthy_Type",
        "CuisineRegion",
        "BestUsdaIngredientName",
    ]

    missing = set(columns_to_keep) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    recipes = {}
    for _, row in df.iterrows():
        recipe_id = row["RecipeId"]
        recipe_data = {col: row[col] for col in columns_to_keep}
        recipes[recipe_id] = recipe_data

    return recipes


# ---------------------------
# Graph Construction Functions
# ---------------------------


def create_graph_and_triples(
    recipes: Dict[Any, Dict[str, Any]],
) -> Tuple[nx.DiGraph, np.ndarray]:
    """
    Build a directed knowledge graph and extract triples from the recipe dictionary.

    The following columns are used:
      - Core recipe node: RecipeId (unique identifier) stored as ("recipe", RecipeId)
      - Attributes modeled as nodes with directed edges from the recipe:
            * Cooking_Method        -> usesCookingMethod
            * Diet_Types            -> hasDietType (list; comma-delimited)
            * meal_type             -> isForMealType (list; comma-delimited)
            * Healthy_Type          -> processed via map_health_attribute
            * CuisineRegion         -> hasCuisineRegion
            * BestUsdaIngredientName -> containsIngredient (list; semicolon-delimited)

    Parameters:
        recipes (Dict[Any, Dict[str, Any]]): Dictionary with RecipeId keys and attribute dictionaries as values.

    Returns:
        Tuple[nx.DiGraph, np.ndarray]: A tuple containing the directed graph and a NumPy array of triples.
    """
    G = nx.DiGraph()
    triples = []

    # Mapping for single-value attributes.
    attribute_mappings = {
        "Cooking_Method": ("usesCookingMethod", "cooking_method"),
        "CuisineRegion": ("hasCuisineRegion", "cuisine_region"),
    }

    # List-based attributes.
    list_attributes = {
        "Diet_Types": ("hasDietType", "diet_type", ","),  # comma-delimited
        "meal_type": ("isForMealType", "meal_type", ","),  # comma-delimited
    }

    # For ingredients, we use only BestUsdaIngredientName.
    ingredient_relation = "containsIngredient"
    ingredient_node_type = "ingredient"
    ingredient_delimiter = ";"  # semicolon-delimited

    for recipe_id, details in recipes.items():
        # Create a recipe node using a tuple identifier.
        recipe_node = ("recipe", recipe_id)
        G.add_node(recipe_node, type="recipe", RecipeId=recipe_id)

        # Process single-value attributes.
        for col, (relation, node_type) in attribute_mappings.items():
            element = details.get(col, None)
            if (
                element
                and element != UNKNOWN_PLACEHOLDER
                and str(element).strip() != ""
            ):
                element_clean = str(element).strip()
                node_id = (node_type, element_clean)
                if not G.has_node(node_id):
                    G.add_node(node_id, type=node_type, label=element_clean)
                G.add_edge(recipe_node, node_id, relation=relation)
                triples.append((str(recipe_node), relation, str(node_id)))

        # Process Healthy_Type with custom mapping.
        healthy = details.get("Healthy_Type", None)
        if healthy and healthy != UNKNOWN_PLACEHOLDER and str(healthy).strip() != "":
            if isinstance(healthy, list):
                healthy_elements = [str(h).strip() for h in healthy if str(h).strip()]
            else:
                healthy_elements = split_and_clean(str(healthy), ",")
            for element in healthy_elements:
                if element:
                    relation = map_health_attribute(element)
                    node_id = ("health_attribute", element)
                    if not G.has_node(node_id):
                        G.add_node(node_id, type="health_attribute", label=element)
                    G.add_edge(recipe_node, node_id, relation=relation)
                    triples.append((str(recipe_node), relation, str(node_id)))

        # Process list-based attributes.
        for col, (relation, node_type, delimiter) in list_attributes.items():
            value = details.get(col, None)
            if value and value != UNKNOWN_PLACEHOLDER and str(value).strip() != "":
                if isinstance(value, list):
                    elements = [str(v).strip() for v in value if str(v).strip()]
                else:
                    elements = split_and_clean(str(value), delimiter)
                for element in elements:
                    if element:
                        node_id = (node_type, element)
                        if not G.has_node(node_id):
                            G.add_node(node_id, type=node_type, label=element)
                        G.add_edge(recipe_node, node_id, relation=relation)
                        triples.append((str(recipe_node), relation, str(node_id)))

        # Process ingredients using only BestUsdaIngredientName.
        best_usda = details.get("BestUsdaIngredientName", None)
        if (
            best_usda
            and best_usda != UNKNOWN_PLACEHOLDER
            and str(best_usda).strip() != ""
        ):
            ingredients = split_and_clean(str(best_usda), ingredient_delimiter)
            for ingredient in ingredients:
                if ingredient:
                    node_id = ("ingredient", ingredient)
                    if not G.has_node(node_id):
                        G.add_node(node_id, type=ingredient_node_type, label=ingredient)
                    G.add_edge(recipe_node, node_id, relation=ingredient_relation)
                    triples.append(
                        (str(recipe_node), ingredient_relation, str(node_id))
                    )

    triples_array = np.array(triples, dtype=str)
    return G, triples_array


def save_triples(triples_array: np.ndarray, file_path: str) -> None:
    """
    Save the triples array to a CSV file.

    Parameters:
        triples_array (np.ndarray): Numpy array of triples (Head, Relation, Tail).
        file_path (str): Destination file path.
    """
    triples_df = pd.DataFrame(triples_array, columns=["Head", "Relation", "Tail"])
    triples_df.to_csv(file_path, index=False)


def save_graph(G: nx.DiGraph, file_path: str) -> None:
    """
    Persist the graph to a file using pickle.

    Parameters:
        G (nx.DiGraph): The knowledge graph.
        file_path (str): Destination file path.
    """
    with open(file_path, "wb") as f:
        pickle.dump(G, f)


# ---------------------------
# Main Execution Block
# ---------------------------

if __name__ == "__main__":
    # Read the CSV file using Pandas.
    csv_path = "/app/data/dataFullLargerRegionAndCountryWithServingsBin.csv"
    df = pd.read_csv(csv_path)

    # Load recipes from the DataFrame; only retain the necessary columns.
    recipes = load_recipes_from_dataframe(df)

    # Create the knowledge graph and extract triples.
    graph, triples = create_graph_and_triples(recipes)

    # Save the extracted triples and the graph.
    save_triples(triples, "/app/train_new_kge_model/triples_new_without_ct_ss.csv")
    save_graph(graph, "/app/train_new_kge_model/knowledge_graph_new_without_ct_ss.pkl")

    # For demonstration, print the first few triples.
    print("Sample triples:")
    print(triples[:5])
