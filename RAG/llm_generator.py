import json
import openai
import os
from dotenv import load_dotenv
from recommender import *

load_dotenv()

def create_messages(user_prompt: str, system_prompt: str = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages

def get_response(model, messages):
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return completion.choices[0].message.content

def get_prompt(user_criteria: dict, MODEL_PATH: Path, TRIPLES_PATH: Path, RECIPE_PATH_CSV: Path) -> str:
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
    # print("---- Prompt Sent to LLM ----")
    # print(prompt)
    # print("---- End of Prompt ----\n")
    
    return prompt

def get_json(raw_text):

    # Remove leading/trailing code fences if any
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        json_text = "\n".join(lines).strip()
    else:
        json_text = raw_text

    if not json_text:
        # fallback JSON if LLM returns nothing
        fallback = {
            "recipes": [
                {
                    "id": "",
                    "title": "No Recipe Generated",
                    "description": "The language model did not return a recipe. Please try again.",
                    "imageUrl": "",
                    "cookingTime": 0,
                    "servings": 0,
                    "calories": 0,
                    "difficulty": "",
                    "categories": [],
                    "cookingMethod": "",
                    "ingredients": [],
                    "steps": [],
                    "rating": 0.0,
                    "reviews": 0,
                    "userId": "",
                    "userName": "",
                    "createdAt": "",
                    "nutritionInfo": {}
                }
            ]
        }
        return json.dumps(fallback)

    return json_text

def get_prompt_evaluation(recipe_text):
    
    system_prompt = """You are a professional chef and culinary critic. 
    Your task is to evaluate AI-generated recipes using the following criteria. 
    Provide only numerical scores from 1 (poor) to 5 (excellent), with no explanation:

- Correctness (are the steps valid and logical?)
- Completeness (does it have all ingredients and instructions?)
- Clarity (are the instructions clear and easy to follow?)
- Creativity (is the recipe unique or imaginative?)
- Feasibility (can it be realistically cooked at home?)
- Estimated Taste (based on known flavor combinations)

Respond in JSON format like:
{
  "Correctness": X,
  "Completeness": X,
  "Clarity": X,
  "Creativity": X,
  "Feasibility": X,
  "Estimated_Taste": X
}
"""

    user_prompt = f"""Here is a recipe generated by an AI assistant:
{recipe_text}
Please evaluate the recipe."""

    return system_prompt, user_prompt

def get_json_evaluation(raw_text):

    # Remove leading/trailing code fences if any
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        json_text = "\n".join(lines).strip()
    else:
        json_text = raw_text

    if not json_text:
        # fallback JSON if LLM returns nothing
        fallback = {
            "evaluation": [
                {
                    "Correctness": "",
                    "Completeness": "",
                    "Clarity": "",
                    "Creativity": "",
                    "Feasibility": "",
                    "Estimated_Taste": ""
                }
            ]
        }
        return json.dumps(fallback)

    return json_text