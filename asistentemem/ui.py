import gradio as gr
import atexit
from asistentemem.speech import speech_to_text
from asistentemem.model import chat_with_ollama, initialize_model, cleanup_model
from asistentemem.data import cargar_documento, obtener_datos_api

# Global variable for TTS state
tts_enabled = False

# Register cleanup function to run when Python exits
atexit.register(cleanup_model)


def update_tts_state(value):
    """Update the global TTS state"""
    global tts_enabled
    tts_enabled = value
    # Return an update to make the audio component visible/invisible based on checkbox state
    return gr.update(visible=value)


def crear_interfaz():
    """Create the Gradio interface"""
    global tts_enabled

    # Initialize the model at startup
    print("Initializing model at application startup...")
    initialize_model()

    with gr.Blocks(
        css=""" 
        .chat-message { border-radius: 15px; padding: 10px; margin: 5px 0; }
        .chatbot { background-color: #f4f4f4; border-radius: 10px; padding: 10px; }
        .user-message { background-color: #cce5ff; color: black; }
        .bot-message { background-color: #d4edda; color: black; }
        body { font-family: Arial, sans-serif; background-color: #f9f9f9; }
        
        /* Audio player styling */
        #component-8:not(.visible) {
            display: none !important;
        }
        
        /* Add any additional custom styles here */
        """
    ) as demo:
        with gr.Row():
            toggle_sidebar_btn = gr.Button("⚙️ Configuración")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Image(
                    "logo.jpg",
                    label="",
                    height=40,
                    width=40,
                    show_label=False,
                    show_download_button=False,
                    interactive=False,
                )
                gr.Markdown("# Asistente mercado de energía Mayorista")
                chatbot = gr.Chatbot(label="🤖 Chatbot", type="messages")
                # NEW: Audio component to play TTS audio on the client browser
                audio_output = gr.Audio(
                    label="Audio respuesta (TTS)",
                    type="filepath",
                    elem_id="audio-output",
                    autoplay=True,
                    visible=tts_enabled,
                )
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="💬 Escribe tu mensaje",
                        placeholder="Escribe algo...",
                        lines=2,
                        scale=8,
                    )
                    voice_input_btn = gr.Button("🎙 Micrófono", scale=2)
                submit_btn = gr.Button("🚀 Enviar")

            with gr.Column(scale=1, visible=False) as sidebar:
                gr.Markdown("📢 **Ajustes del Modelo**")
                with gr.Accordion("⚙️ Configuración Avanzada", open=False):
                    gr.Markdown("""
                    *Ajustes que controlan el comportamiento del modelo:*
                    - **Top K/P**: Controla la diversidad de palabras. Valores bajos = respuestas más directas
                    - **Temperatura**: Alta = más creatividad, Baja = más precisión
                    - **Tokens máximos**: Controla la longitud de la respuesta
                    """)
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=50, step=1, value=20, label="🔍 Top K"
                    )
                    top_p_slider = gr.Slider(
                        minimum=0, maximum=1, step=0.01, value=0.7, label="🎯 Top P"
                    )
                    temperatura_slider = gr.Slider(
                        minimum=0,
                        maximum=2,
                        step=0.01,
                        value=0.5,
                        label="🔥 Temperatura",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=20,
                        maximum=200,
                        step=10,
                        value=50,
                        label="📏 Tokens máximos",
                    )

                tts_checkbox = gr.Checkbox(
                    label="🗣 Activar TTS (Text-to-Speech)",
                    value=False,
                    elem_id="tts-checkbox",
                )
                file_upload = gr.File(label="📂 Subir documento .docx", type="filepath")
                gr.Markdown("### 🌟 La Columna no puede superar dos años de consulta")
                api_input = gr.Textbox(
                    label="🌐 URL de API (Opcional)",
                    placeholder="Introduce una API para el chat",
                )

                # Add unload model button at the bottom of sidebar
                unload_model_btn = gr.Button("🧹 Liberar recursos del modelo")
                unload_model_btn.click(fn=cleanup_model, inputs=[], outputs=[])

        sidebar_state = gr.State(False)

        toggle_sidebar_btn.click(
            fn=lambda state: (not state, gr.update(visible=not state)),
            inputs=[sidebar_state],
            outputs=[sidebar_state, sidebar],
        )

        voice_input_btn.click(
            speech_to_text, inputs=[], outputs=prompt_input, show_progress=True
        )
        file_upload.change(cargar_documento, inputs=[file_upload], outputs=chatbot)
        api_input.change(obtener_datos_api, inputs=[api_input], outputs=chatbot)

        # Update submit_btn to output both chatbot messages and an audio file (if any)
        submit_btn.click(
            chat_with_ollama,
            inputs=[
                prompt_input,
                top_k_slider,
                top_p_slider,
                temperatura_slider,
                max_tokens_slider,
                tts_checkbox,
            ],
            outputs=[chatbot, audio_output],
        )

        # Add an event handler to update the global tts_enabled variable
        tts_checkbox.change(
            fn=update_tts_state, inputs=[tts_checkbox], outputs=[audio_output]
        )

    return demo
