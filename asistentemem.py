import gradio as gr
import os
from asistentemem.ui import crear_interfaz

# Create the directory structure if it doesn't exist
os.makedirs("asistentemem", exist_ok=True)

# Import modules

# Main application
if __name__ == "__main__":
    demo = crear_interfaz()
    demo.launch(share=True)
