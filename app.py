from flask import Flask, render_template, request, jsonify
from openai import OpenAI, OpenAIError
from flask_sqlalchemy import SQLAlchemy
import logging
import traceback
import requests
from datetime import datetime
from PIL import Image
import os
from io import BytesIO
import uuid

app = Flask(__name__)

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# API Keys and Configuration
API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyBURcqLSjTUq2lKNKeIcAQfhzkxhYy0bGI"
SEARCH_ENGINE_ID = "10538e636f7334631"

# Directory to save generated images
GENERATED_IMAGES_DIR = os.path.join(app.root_path, 'static', 'generated_images')
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///conversation_history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Model
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(50), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

# Create Database Tables
with app.app_context():
    db.create_all()

# Initialize OpenAI Client
try:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=API_KEY
    )
    if not client.api_key:
        logging.error("API key is not set.")
        raise ValueError("Missing API Key")
except Exception as e:
    logging.error("Failed to initialize OpenAI client.", exc_info=True)
    raise e

# Function to Perform Live Search
def perform_live_search(query, max_results=3):
    """
    Performs a live search using Google Custom Search API.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of search results to retrieve.

    Returns:
        list: A list of search results with title, link, and snippet.
    """
    search_endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": max_results
    }

    try:
        response = requests.get(search_endpoint, params=params)
        response.raise_for_status()
        search_results = response.json()
        results = []
        for item in search_results.get("items", []):
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
        logging.debug(f"Live search results for query '{query}': {results}")
        return results
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred during live search: {http_err}")
    except Exception as e:
        logging.error("An error occurred during live search.", exc_info=True)
    return []

# Function to Generate and Process Image
def generate_and_process_image(prompt):
    """
    Generates an image based on the prompt using Pollinations API,
    clips the bottom 50 pixels, saves it, and returns the image URL.

    Args:
        prompt (str): The image generation prompt.

    Returns:
        str: The URL to the processed image.
    """
    try:
        # Generate Image URL
        image_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
        logging.debug(f"Generated image URL: {image_url}")

        # Fetch the image
        response = requests.get(image_url)
        response.raise_for_status()

        # Open the image
        img = Image.open(BytesIO(response.content))
        logging.debug(f"Original image size: {img.size}")

        # Clip the bottom 50 pixels
        width, height = img.size
        if height <= 50:
            logging.error("Image height is too small to clip.")
            return None
        img_clipped = img.crop((0, 0, width, height - 50))
        logging.debug(f"Clipped image size: {img_clipped.size}")

        # Save the processed image
        unique_filename = f"{uuid.uuid4().hex}.png"
        save_path = os.path.join(GENERATED_IMAGES_DIR, unique_filename)
        img_clipped.save(save_path)
        logging.debug(f"Processed image saved at: {save_path}")

        # Return the image URL
        image_url_static = f"/static/generated_images/{unique_filename}"
        return image_url_static

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred during image generation: {http_err}")
    except Exception as e:
        logging.error("An error occurred during image generation and processing.", exc_info=True)
    return None

# Route for Home Page
@app.route("/")
def index():
    return render_template("index.html")

# Chat Route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    model = data.get("model", "default")  # Default to 'default' model if not specified

    if not prompt:
        logging.warning("No prompt provided in the request.")
        return jsonify({"error": "No prompt provided"}), 400

    # **Modified Section Start**
    if prompt.lower().startswith('/image'):
        # Handle /image command
        image_prompt = prompt[6:].strip()
        if not image_prompt:
            logging.warning("Image command used without a prompt.")
            return jsonify({"error": "Please provide a prompt after '/image'."}), 400

        image_url = generate_and_process_image(image_prompt)
        if image_url:
            assistant_message = Message(role="assistant", content=f"<img src='{image_url}' alt='Generated Image' />")
            db.session.add(assistant_message)
            db.session.commit()

            # Fetch the updated last 10 messages to return in the response
            updated_conversation = Message.query.order_by(Message.timestamp.desc()).limit(10).all()
            updated_conversation = list(reversed(updated_conversation))
            updated_conversation_dict = [msg.to_dict() for msg in updated_conversation]

            return jsonify({"response": f"<img src='{image_url}' alt='Generated Image' />", "conversation": updated_conversation_dict })
        else:
            return jsonify({"error": "Failed to generate image."}), 500
    # **Modified Section End**

    # **Existing Handling for /search and regular prompts**
    # Retrieve the last 10 messages from the database ordered by timestamp descending
    last_10_messages = Message.query.order_by(Message.timestamp.desc()).limit(10).all()
    # Reverse the list to maintain chronological order
    conversation_history = reversed(last_10_messages)

    cleaned_history = []
    for msg in conversation_history:
        cleaned_history.append({
            "role": msg.role,
            "content": msg.content
        })

    if prompt.lower().startswith('/search'):
        search_query = prompt[7:].strip()
        if not search_query:
            logging.warning("Search command used without a query.")
            return jsonify({"error": "Please provide a search query after '/search'."}), 400

        search_results = perform_live_search(search_query, max_results=3)

        if search_results:
            search_info = "Here are some recent search results:\n"
            for idx, result in enumerate(search_results, 1):
                search_info += f"{idx}. **{result['title']}**\n{result['snippet']}\nURL: {result['link']}\n\n"
            prompt_with_search = f"{search_query}\n\n{search_info}"
        else:
            prompt_with_search = f"{search_query}\n\nNo search results found."

        cleaned_history.append({"role": "user", "content": prompt_with_search })

        user_message = Message(role="user", content=prompt_with_search)
        db.session.add(user_message)
        db.session.commit()

    else:
        prompt_with_search = prompt

        cleaned_history.append({"role": "user", "content": prompt_with_search })

        user_message = Message(role="user", content=prompt_with_search)
        db.session.add(user_message)
        db.session.commit()

    logging.debug(f"Final conversation_history: {cleaned_history}")

    try:
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=cleaned_history,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=False
        )

        ai_response = response.choices[0].message.content.strip()
        logging.debug(f"AI Response: {ai_response}")

        assistant_message = Message(role="assistant", content=ai_response)
        db.session.add(assistant_message)
        db.session.commit()

        # **Modified Section Start**
        # Fetch the updated last 10 messages to return in the response
        updated_conversation = Message.query.order_by(Message.timestamp.desc()).limit(10).all()
        # Reverse to chronological order
        updated_conversation = list(reversed(updated_conversation))
        # Convert to dictionaries
        updated_conversation_dict = [msg.to_dict() for msg in updated_conversation]
        # **Modified Section End**

        return jsonify({"response": ai_response, "conversation": updated_conversation_dict })

    except OpenAIError as oe:
        logging.error("OpenAI API returned an error.", exc_info=True)
        return jsonify({"error": f"OpenAI API error: {str(oe)}"}), 500
    except Exception as e:
        logging.error("An unexpected error occurred.", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
