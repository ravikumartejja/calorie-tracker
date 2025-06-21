import os
import io
import base64
import json
import requests # For making HTTP requests to Gemini API
from PIL import Image
import gradio as gr # Import Gradio

# --- 1. Configuration and Setup ---

# Retrieve Google Gemini API key from environment variables (Hugging Face Space Secrets)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # This message will appear if the secret isn't set on Hugging Face
    # or if running locally without a .env file.
    print("Error: Google Gemini API key (GEMINI_API_KEY) is not set.")
    print("Please set it as a Space Secret on Hugging Face or in a .env file locally.")
    # In a deployed Gradio app, we might return an error message to the user
    # rather than exiting the script.
    
# Define the Gemini model to use
GEMINI_MODEL_NAME = "gemini-1.5-flash" # Changed to a commonly available model for broader access, you can adjust
                                    # based on your Gemini account access and rate limits.

# Gemini API endpoint
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"


# --- 2. Image Processing Functions ---

def process_image_for_analysis(image, max_size=(800, 800), quality=85):
    """
    Processes image by resizing and compressing while maintaining quality.
    Args:
        image: PIL Image object
        max_size: tuple of maximum dimensions (width, height)
        quality: JPEG compression quality (0-100)
    Returns:
        bytes of the processed image
    """
    try:
        img_copy = image.copy()
        img_copy.thumbnail(max_size, Image.LANCZOS)
        image_bytes_io = io.BytesIO()
        img_copy.save(image_bytes_io, format='JPEG', quality=quality)
        return image_bytes_io.getvalue()
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

# --- 3. Nutritional Response Parsing ---

def parse_nutrition_response(response_text):
    """
    Parse the JSON response from the model.
    Adjusted to be more robust to potential non-JSON output or partial JSON.
    """
    try:
        # Attempt to find the JSON block if the model wraps it in markdown (common for LLMs)
        cleaned_response = response_text.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(cleaned_response)
        
        # Safely get data with .get() and provide defaults
        macros = data.get('macronutrients', {})
        micros = data.get('micronutrients', {})
        additional_info = data.get('additional_info', {})
        improvements = data.get('improvements', {'suggestions': [], 'context': ''})
        
        return {
            'food_items': data.get('identified_foods', []),
            'macronutrients': macros,
            'micronutrients': micros,
            'improvements': improvements,
            'additional_info': additional_info
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from model response: {str(e)}")
        print(f"Raw response from model (check for malformed JSON): \n{response_text}")
        return {
            'food_items': [],
            'macronutrients': {},
            'micronutrients': {},
            'improvements': {'suggestions': ["Could not parse response. Please try again or refine input."], 'context': ''},
            'additional_info': {}
        }
    except Exception as e:
        print(f"An unexpected error occurred parsing nutrition data: {str(e)}")
        return {
            'food_items': [],
            'macronutrients': {},
            'micronutrients': {},
            'improvements': {'suggestions': ["An unexpected error occurred during parsing."], 'context': ''},
            'additional_info': {}
        }

# --- 4. Image Analysis with Gemini API ---

def analyze_image_with_gemini_api(api_url, image_bytes, user_goal="Maintain weight"):
    """
    Analyzes an image using the Google Gemini API.
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # The prompt should be concise and guide the model to output JSON
    prompt_text = f"""
    Analyze the food items in this image and provide the nutritional information in the following JSON format only.
    Consider the user's goal: '{user_goal}' when suggesting improvements.
    
    {{
        "identified_foods": [
            "food item 1",
            "food item 2"
        ],
        "macronutrients": {{
            "carbohydrates": number,
            "protein": number,
            "fat": number,
            "calories": number,
            "sugar": number,
            "saturated_fat": number,
            "cholesterol": number,
            "sodium": number
        }},
        "micronutrients": {{
            "vitamin_a": number,
            "vitamin_c": number,
            "calcium": number,
            "iron": number,
            "fiber": number,
            "vitamin_d": number,
            "vitamin_e": number,
            "vitamin_k": number,
            "thiamin": number,
            "riboflavin": number,
            "niacin": number,
            "vitamin_b6": number,
            "folate": number,
            "vitamin_b12": number,
            "pantothenic_acid": number,
            "potassium": number,
            "magnesium": number,
            "zinc": number,
            "selenium": number,
            "copper": number,
            "manganese": number
        }},
        "improvements": {{
            "suggestions": [
                "🌟 Great choice on including [positive aspect]!",
                "💪 Keep up the good work with [healthy element]!",
                "💡 Consider adding [suggestion] to boost nutrition"
            ],
            "context": "Start with encouraging feedback about the healthy aspects of the meal, then provide constructive suggestions. Use emojis like 🥗 for healthy choices, 💪 for protein-rich foods, 🌟 for balanced meals, 🍎 for fruits/vegetables, 💚 for nutritious choices. Be concise."
        }},
        "additional_info": {{
            "serving_size": "e.g., 200g or 1 cup",
            "total_weight": "e.g., 350g",
            "dietary_restrictions": "e.g., Gluten-free, Vegan",
            "allergens": "e.g., Peanuts, Dairy"
        }}
    }}

    If you cannot identify a specific value, set it to 0.
    Provide ONLY the JSON response without any additional text or explanation before or after the JSON.
    """

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg", # Ensure this matches your image format
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1, 
            "topP": 1.0,
            "topK": 0,
            "responseMimeType": "application/json" # Explicitly request JSON
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()
        
        if result.get('candidates') and result['candidates'][0].get('content') and \
           result['candidates'][0]['content'].get('parts') and result['candidates'][0]['content']['parts'][0].get('text'):
            return result['candidates'][0]['content']['parts'][0]['text']
        elif result.get('error'):
            return f"Gemini API Error: {result['error'].get('message', 'Unknown API error')}"
        else:
            return f"Unexpected Gemini API response structure: {result}"

    except requests.exceptions.RequestException as e:
        return f"Network or request error during Gemini API call: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON response from Gemini API: {e}"
    except Exception as e:
        return f"An unexpected error occurred during Gemini API inference: {e}"

# --- Gradio Interface Function ---

def get_nutritional_info(image: Image.Image, user_goal: str):
    if GEMINI_API_KEY is None:
        return "Error: Gemini API Key is not configured. Please set it as a Hugging Face Space Secret."

    processed_image_bytes = process_image_for_analysis(image)
    if processed_image_bytes is None:
        return "Failed to process image. Please try another image."
        
    raw_response_text = analyze_image_with_gemini_api(GEMINI_API_URL, processed_image_bytes, user_goal)
    
    if not raw_response_text or "Error:" in raw_response_text:
        return raw_response_text # Propagate API errors

    parsed_result = parse_nutrition_response(raw_response_text)
    
    # Format the output for Gradio display
    output_text = "### Nutritional Analysis\n\n"
    
    output_text += "**Identified Foods:**\n"
    if parsed_result['food_items']:
        for food in parsed_result['food_items']:
            output_text += f"- {food}\n"
    else:
        output_text += "No specific food items identified.\n"

    output_text += "\n**Macronutrients:**\n"
    macros = parsed_result.get('macronutrients', {})
    if macros:
        for macro, value in macros.items():
            macro_display_name = macro.replace('_', ' ').title()
            if macro == 'calories':
                output_text += f"- {macro_display_name}: {value} kcal\n"
            elif macro in ['sodium', 'cholesterol']:
                output_text += f"- {macro_display_name}: {value} mg\n"
            else:
                output_text += f"- {macro_display_name}: {value}g\n"
    else:
        output_text += "No macronutrient data available.\n"
    
    output_text += "\n**Micronutrients:**\n"
    micros = parsed_result.get('micronutrients', {})
    if micros:
        for micro, value in micros.items():
            micro_display_name = micro.replace('_', ' ').title()
            if micro == 'vitamin_a':
                output_text += f"- {micro_display_name}: {value} IU\n"
            elif micro == 'fiber':
                output_text += f"- {micro_display_name}: {value}g\n"
            else:
                output_text += f"- {micro_display_name}: {value} mg\n"
    else:
        output_text += "No micronutrient data available.\n"

    additional_info = parsed_result.get('additional_info', {})
    if additional_info:
        output_text += "\n**Additional Information:**\n"
        for key, value in additional_info.items():
            output_text += f"- {key.replace('_', ' ').title()}: {value}\n"
            
    output_text += "\n**Suggested Improvements:**\n"
    output_text += f"Based on your goal: {user_goal}\n"
    improvements = parsed_result.get('improvements', {})
    if improvements and improvements.get('suggestions'):
        for suggestion in improvements['suggestions']:
            output_text += f"- {suggestion}\n"
        if improvements.get('context'):
            output_text += f"\nContext: {improvements['context']}\n"
    else:
        output_text += "No specific suggestions provided.\n"

    return output_text

# --- Gradio Interface Definition ---

goals_list = [
    "Maintain weight", "Fat loss", "Weight gain", "Muscle Gain",
    "Pregnancy", "Body Building Competition", "Marathon Training",
    "Endurance Training", "Senior Citizen", "Diabetic Patient",
    "Kidney Patient"
]

demo = gr.Interface(
    fn=get_nutritional_info,
    inputs=[
        gr.Image(type="pil", label="Upload Food Image"),
        gr.Dropdown(goals_list, label="Select Your Goal", value="Maintain weight")
    ],
    outputs=gr.Markdown(label="Nutritional Analysis Results"),
    title="AI Food Calorie and Nutrition Tracker",
    description="Upload a picture of your food, select your health goal, and get detailed nutritional information and suggestions powered by Google Gemini.",
    examples=[
        # You can add example image paths here if you upload them to your space
        # For instance, if you upload an image named 'salad.jpg' to your space's root:
        # ["salad.jpg", "Maintain weight"]
    ]
)

if __name__ == "__main__":
    demo.launch() # This will launch the Gradio app when run locally