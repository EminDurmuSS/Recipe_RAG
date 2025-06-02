from recommender import *

def generate_recipe_llm(user_criteria: dict, MODEL_PATH: Path, TRIPLES_PATH: Path, RECIPE_PATH_CSV: Path) -> str:
    """
    Generate a new recipe using the LLM based on user criteria + 5 example recipes for inspiration.
    Returns a JSON string containing { "recipes": [ {...}, ... ] }.
    """
    criteria = map_user_input_to_criteria(
        cooking_method=user_criteria.get("cooking_method", ""),
        diet_types=user_criteria.get("diet_types", []),
        meal_type=user_criteria.get("meal_type", []),
        health_types=user_criteria.get("health_types", []),
        cuisine_region=user_criteria.get("cuisine_region", ""),
        ingredients=user_criteria.get("ingredients", []),
        weights=user_criteria.get("weights", {})
    )

    # Retrieve up to 5 sample recipe IDs using the KGE model
    example_recipe_ids = get_matching_recipes(criteria=criteria,
                                              MODEL_PATH=MODEL_PATH,
                                              TRIPLES_PATH=TRIPLES_PATH,
                                              top_k=5,
                                              flexible=False)
    
    # Fetch detailed info for the sample recipes
    example_recipes = []
    for rid in example_recipe_ids:
        recipe_info = fetch_recipe_info(rid, RECIPE_PATH_CSV)
        if recipe_info:
            example_recipes.append(recipe_info)

    # Format the example recipes for the prompt
    formatted_examples = ""
    for idx, recipe in enumerate(example_recipes, start=1):
        formatted_examples += f"Example Recipe {idx}:\n{format_recipe_example(recipe)}\n{'-'*40}\n"

    formatted_criteria = format_user_criteria(user_criteria)

    # Construct final LLM prompt
    prompt = f"""
You are a world-class culinary expert and innovative chef known for your exceptionally creative dishes.
Your task is to generate a completely new recipe that meets the following criteria:

{formatted_criteria}

Use the following example recipes for inspiration:
{formatted_examples}

Output a single JSON object with a top-level key "recipes". This key should map
to an array containing exactly one recipe object with these keys (and no extra keys):

- "id": A unique identifier (can be empty)
- "title": A creative and enticing recipe title
- "description": A vivid and imaginative description 
- "imageUrl": The final image URL of the dish
- "cookingTime": Total cooking time (minutes)
- "servings": Number of servings
- "calories": Calorie count
- "difficulty": Recipe difficulty (Easy, Medium, Hard)
- "categories": An array of category strings
- "cookingMethod": The cooking method used
- "ingredients": An array of ingredient objects
- "steps": An array of instruction objects
- "nutritionInfo": An object containing nutritional details

Example JSON:
{{
  "recipes": [
    {{
      "id": "",
      "title": "Example Recipe Title",
      "description": "A brief creative description incorporating.",
      "cookingTime": 45,
      "servings": 4,
      "calories": 500,
      "difficulty": "Medium",
      "categories": ["Example", "Inspiration"],
      "cookingMethod": "baking",
      "ingredients": [
        {{
          "name": "Ingredient 1",
          "amount": "1",
          "unit": "cup",
          "notes": "",
          "category": "Category",
          "amountInGrams": 100
        }}
      ],
      "steps": [
        {{
          "title": "Step 1",
          "description": "Do something.",
          "duration": 10
        }}
      ],
      "nutritionInfo": {{
        "calories": 500,
        "protein": 20,
        "carbohydrates": 50,
        "fat": 10,
        "saturatedFat": 2,
        "transFat": 0,
        "cholesterol": 0,
        "sodium": 300,
        "fiber": 5,
        "sugars": 8,
        "vitaminD": 0,
        "calcium": 100,
        "iron": 5,
        "potassium": 400,
        "fatDailyValue": 15,
        "saturatedFatDailyValue": 10,
        "cholesterolDailyValue": 0,
        "sodiumDailyValue": 13,
        "carbohydratesDailyValue": 20,
        "fiberDailyValue": 25,
        "proteinDailyValue": 30,
        "vitaminDDailyValue": 0,
        "calciumDailyValue": 10,
        "ironDailyValue": 15,
        "potassiumDailyValue": 12,
        "servingSize": "1 serving"
      }}
    }}
  ]
}}
Now, please generate the recipe.
"""
    print("---- Prompt Sent to Gemini LLM ----")
    print(prompt)
    print("---- End of Prompt ----\n")
    
    return prompt