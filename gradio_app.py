import pickle
import gradio as gr
from TranslationModel import TranslationModel

# Load the tokenizer
with open("eng_tokenizer.pkl", "rb") as f:
    eng_tokenizer = pickle.load(f)

# Initialize the translation model
model = TranslationModel("final_model.keras", "eng_tokenizer.pkl", max_len=10)

def translate_sentence(sentence):
    return model.predict(sentence)

# Create a Gradio interface
iface = gr.Interface(fn=translate_sentence, inputs="text", outputs="text")

# Launch the interface
iface.launch()
