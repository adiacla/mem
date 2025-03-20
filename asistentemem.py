import gradio as gr
import os
from asistentemem.ui import crear_interfaz


# Import modules

# Main application
if __name__ == "__main__":
    demo = crear_interfaz()
    demo.launch(share=True)