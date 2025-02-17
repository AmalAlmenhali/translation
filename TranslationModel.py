import logging
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class TranslationModel:
    def __init__(self, model_path, eng_tokenizer_path, fr_tokenizer_path, max_len):
        logging.info("TranslationModel class initialized")

        # Load the trained model
        self.model = load_model(model_path)
        self.max_len = max_len

        # Load the English tokenizer
        with open(eng_tokenizer_path, "rb") as f:
            self.eng_tokenizer = pickle.load(f)

        # Load the French tokenizer
        with open(fr_tokenizer_path, "rb") as f:
            self.fr_tokenizer = pickle.load(f)

        logging.info("Model and both tokenizers are loaded and compiled!")

    def predict(self, sentence):
        """Predict translation for a given English sentence."""
        sentence_seq = self.eng_tokenizer.texts_to_sequences([sentence])
        sentence_padded = pad_sequences(sentence_seq, maxlen=self.max_len, padding='post')

        # Predict translation
        prediction = self.model.predict(sentence_padded)

        # Convert predicted sequence to words using French tokenizer
        translated_words = self.decode_translation(prediction)

        return " ".join(translated_words)

    def decode_translation(self, prediction):
        """Convert predicted token indices to words."""
        y_id_to_word = {value: key for key, value in self.fr_tokenizer.word_index.items()}
        y_id_to_word[0] = "<PAD>"

        return [y_id_to_word.get(np.argmax(x), "") for x in prediction[0]]

# Example usage
def main():
    # Initialize the translation model
    model = TranslationModel("final_model.keras", "eng_tokenizer.pkl", "fr_tokenizer.pkl", max_len=10)  # Set correct max_len

    # Test the model
    sentence = "she is driving the truck"
    translation = model.predict(sentence)

    logging.info(f"Translated Sentence: {translation}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
