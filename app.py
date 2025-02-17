import os
import logging
import tensorflow as tf
from flask import Flask, request, jsonify
from TranslationModel import TranslationModel  # Import the translation model

app = Flask(__name__)

# Define model path
model_path = './final_model.keras'

# Load the tokenizer (ensure you have the tokenizer in your project)
from tensorflow.keras.preprocessing.text import Tokenizer

eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(["dummy text"])  # Replace with actual tokenizer loading

# Set the max length of input sequences (same as training)
max_len = 10  # Update this to match your model training

# Create model instance
model = TranslationModel(model_path, eng_tokenizer, max_len)
logging.basicConfig(level=logging.INFO)


@app.route("/")
def index():
    """Provide simple health check route."""
    return "Translation API is running!"


@app.route("/v1/translate", methods=["GET", "POST"])
def translate():
    """Provide translation API route. Responds to both GET and POST requests."""
    logging.info("Translation request received!")

    # Get the input sentence from query parameters or JSON
    if request.method == "GET":
        sentence = request.args.get("sentence")
    elif request.method == "POST":
        sentence = request.json.get("sentence")

    if not sentence:
        return jsonify({"error": "No input sentence provided!"}), 400

    # Get translation
    translation = model.predict(sentence)

    logging.info(f"Predicted translation: {translation}")
    return jsonify({"translated_sentence": translation})


def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
