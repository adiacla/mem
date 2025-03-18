import gradio as gr
import requests
import json
import pyttsx3
import speech_recognition as sr
from docx import Document
import re
import pandas as pd
from gtts import gTTS  # NEW import

document_text = ""
api_text = ""  # Variable para almacenar la respuesta de la API
enable_tts = False  # Variable global para controlar TTS
tts_enabled = False  # Global variable for TTS state


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        recognizer.pause_threshold = 1

        print("🎙 Escuchando...")
        try:
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=10)
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())

            text = recognizer.recognize_google(audio, language="es-ES")
            return text
        except sr.UnknownValueError:
            return "No se pudo entender el audio."
        except sr.RequestError:
            return "Error con el servicio de reconocimiento."
        except sr.WaitTimeoutError:
            return "No se detectó ninguna voz, intenta de nuevo."


# Comment out or remove the old TTS implementation
# def text_to_speech(text):
#     global enable_tts
#     if not enable_tts:
#         return
#     engine = pyttsx3.init()
#     clean_text = re.sub(r"[*_`~]", "", text)
#     engine.say(clean_text)
#     engine.runAndWait()


# NEW function: Generate TTS audio using gTTS and save as an MP3 file.
def generate_tts_audio(text):
    tts = gTTS(text=text, lang="es")
    audio_file = "response.mp3"
    tts.save(audio_file)
    return audio_file


def obtener_datos_api(url):
    global api_text
    try:
        response = requests.get(url)
        print(f"📡 API solicitada: {url}")
        print(f"📡 Código de respuesta: {response.status_code}")

        if response.status_code == 200:
            api_data = response.json()
            # print(f"📡 Respuesta API (cruda): {json.dumps(api_data, indent=2)}")  # Imprime el JSON completo

            if "result" in api_data and "records" in api_data["result"]:
                records = api_data["result"]["records"]

                df = pd.DataFrame(records)

                # Convertir la columna "Fecha" a datetime
                df["Fecha"] = pd.to_datetime(df["Fecha"])

                # Pivotear la tabla: "CodigoVariable" serán las columnas, "Valor" el contenido
                df_pivot = df.pivot(
                    index="Fecha", columns="CodigoVariable", values="Valor"
                ).reset_index()

                # Cambiar el formato de la fecha a "DD-MM-YYYY"
                df_pivot["Fecha"] = df_pivot["Fecha"].dt.strftime("%d-%m-%Y")

                df_pivot.index.name = "index"

                # Convertir el DataFrame a texto legible
                pivot_text = df_pivot.to_string()

                name = api_data["result"].get("name", "Sin nombre")
                description = (
                    api_data["result"]
                    .get("metadata", {})
                    .get("description", "Sin descripción")
                )

                api_text = (
                    f"🔹 **{name}**\n"
                    f"📄 {description}\n\n"
                    f"🌐 **Información cargada desde la API:** {url}\n\n"
                    f"📊 **Conjuntos de dato del API:**\n```\n{pivot_text}\n```"
                )

                print(f"✅ API cargada: {api_text}")
                return [
                    {
                        "role": "assistant",
                        "content": "📄 API cargada exitosamente. ¿Qué desea preguntar?",
                    }
                ]
            else:
                print("❌ Error: No se encontraron registros en la API.")
                return [
                    {
                        "role": "assistant",
                        "content": "❌ Error: No se encontraron registros en la API.",
                    }
                ]
        else:
            print(f"❌ Error en la API: {response.status_code}")
            return [
                {
                    "role": "assistant",
                    "content": f"❌ Error en la API: {response.status_code}",
                }
            ]
    except Exception as e:
        print(f"❌ Error al conectar con la API: {str(e)}")
        return [
            {
                "role": "assistant",
                "content": f"❌ Error al conectar con la API: {str(e)}",
            }
        ]


def chat_with_ollama(prompt, top_k, top_p, temperatura, max_tokens, tts_enabled):
    global document_text, enable_tts, api_text
    enable_tts = tts_enabled

    if not prompt.strip():
        return [
            {
                "role": "assistant",
                "content": "Documento subido correctamente. ¿Qué desea preguntar?",
            }
        ], None

    full_prompt = (
        f"{document_text}\n\n{api_text}\n\nUsuario: {prompt}"
        if document_text or api_text
        else prompt
    )
    url = "http://127.0.0.1:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3.2:3b-instruct-q6_K",
        "messages": [{"role": "user", "content": full_prompt}],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperatura,
        "stream": False,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        assistant_message = response_data["message"]["content"]
        audio_path = generate_tts_audio(assistant_message) if tts_enabled else None
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_message},
        ], audio_path
    else:
        return [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": f"Error: {response.status_code}, {response.text}",
            },
        ], None


def cargar_documento(archivo):
    global document_text
    doc = Document(archivo.name)
    document_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return [
        {
            "role": "assistant",
            "content": "📄 Documento cargado exitosamente. ¿Qué desea preguntar?",
        }
    ]


def crear_interfaz():
    global tts_enabled

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
                    voice_input_btn = gr.Button("🎙 Hablar", scale=2)
                submit_btn = gr.Button("🚀 Enviar")

            with gr.Column(scale=1, visible=False) as sidebar:
                gr.Markdown("📢 **Ajustes del Modelo**")
                with gr.Accordion("⚙️ Configuración Avanzada", open=False):
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
                gr.Markdown("# 🌟 La Columna no puede superar dos años de consulta")
                api_input = gr.Textbox(
                    label="🌐 URL de API (Opcional)",
                    placeholder="Introduce una API para el chat",
                )

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

        # Add JavaScript to handle default playback rate and visibility
        demo.load(
            js="""
        function setupAudio() {
            // Set default playback rate when audio element is available
            const setupPlaybackRate = setInterval(() => {
                const audioElement = document.querySelector('#component-8 audio');
                if (audioElement) {
                    audioElement.defaultPlaybackRate = 1.25;
                    audioElement.playbackRate = 1.25;
                    clearInterval(setupPlaybackRate);
                }
            }, 100);
            
            // Handle visibility based on checkbox state
            const ttsCheckbox = document.querySelector('#tts-checkbox input[type="checkbox"]');
            if (ttsCheckbox) {
                const audioContainer = document.querySelector('#component-8');
                
                // Set initial visibility
                if (audioContainer) {
                    audioContainer.style.display = ttsCheckbox.checked ? 'block' : 'none';
                }
                
                // Add change event listener
                ttsCheckbox.addEventListener('change', (event) => {
                    if (audioContainer) {
                        audioContainer.style.display = event.target.checked ? 'block' : 'none';
                    }
                });
            }
        }
        
        // Run setup when DOM is loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupAudio);
        } else {
            setupAudio();
        }
        """
        )

    return demo


# New function to update the global TTS state
def update_tts_state(value):
    global tts_enabled
    tts_enabled = value
    # Return an update to make the audio component visible/invisible based on checkbox state
    return gr.update(visible=value)


demo = crear_interfaz()
demo.launch(share=True)
