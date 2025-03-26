import gradio as gr
import os
from asistentemem.ui import crear_interfaz
from asistentemem import model  # import the model to set the API choice


preferred_api = "ollamssa"  # SI es  "ollama" usara ollama, si es cualquier otro valor, usara transformers

if preferred_api.lower() == "ollama":
    model.USE_OLLAMA_API = True
else:
    model.USE_OLLAMA_API = False

# Create the directory structure if it doesn't exist
os.makedirs("asistentemem", exist_ok=True)

# Main application
if __name__ == "__main__":
    demo = crear_interfaz()
    demo.launch(share=True)
