import gradio as gr
import requests
import json
import pyttsx3
import speech_recognition as sr
from docx import Document
import re

document_text = ""
enable_tts = False  # Variable global para controlar TTS

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

def text_to_speech(text):
    global enable_tts
    if not enable_tts:
        return  # Evita la conversión si el usuario la desactivó

    engine = pyttsx3.init()
    clean_text = re.sub(r'[*_`~]', '', text)
    engine.say(clean_text)
    engine.runAndWait()

def chat_with_ollama(prompt, top_k, top_p, temperatura, max_tokens, tts_enabled):
    global document_text, enable_tts
    enable_tts = tts_enabled  # Actualiza el estado de TTS
    
    if not prompt.strip():
        return [["", "Documento subido correctamente. ¿Qué desea preguntar?"]]
    
    full_prompt = f"{document_text}\n\nUsuario: {prompt}" if document_text else prompt
    url = "http://127.0.0.1:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3.2:3b-instruct-q6_K",
        "messages": [{"role": "user", "content": full_prompt}],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperatura,
        "stream": False,
        "max_tokens": max_tokens
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        assistant_message = response_data['message']['content']
        text_to_speech(assistant_message)
        return [[prompt, assistant_message]]
    else:
        return [[prompt, f"Error: {response.status_code}, {response.text}"]]

def cargar_documento(archivo):
    global document_text
    doc = Document(archivo.name)
    document_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return [["", "📄 Documento cargado exitosamente. ¿Qué desea preguntar?"]]

def crear_interfaz():
    with gr.Blocks(css=""" 
        .chat-message { border-radius: 15px; padding: 10px; margin: 5px 0; }
        .chatbot { background-color: #f4f4f4; border-radius: 10px; padding: 10px; }
        .user-message { background-color: #cce5ff; color: black; }
        .bot-message { background-color: #d4edda; color: black; }
        body { font-family: Arial, sans-serif; background-color: #f9f9f9; }
    """) as demo:
        with gr.Row():
            toggle_sidebar_btn = gr.Button("⚙️ Configuración")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# 🌟 Chat con Ollama 3.2 con Voz y Texto")
                chatbot = gr.Chatbot(label="🤖 Chatbot")        
                with gr.Row():
                    prompt_input = gr.Textbox(label="💬 Escribe tu mensaje", placeholder="Escribe algo...", lines=2, scale=8)
                    voice_input_btn = gr.Button("🎙 Hablar", scale=2)       
                submit_btn = gr.Button("🚀 Enviar")
            
            with gr.Column(scale=1, visible=False) as sidebar:
                gr.Markdown("📢 **Ajustes del Modelo**")
                with gr.Accordion("⚙️ Configuración Avanzada", open=False):
                    top_k_slider = gr.Slider(minimum=1, maximum=50, step=1, value=20, label="🔍 Top K")
                    top_p_slider = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.7, label="🎯 Top P")
                    temperatura_slider = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.5, label="🔥 Temperatura")
                    max_tokens_slider = gr.Slider(minimum=20, maximum=200, step=10, value=50, label="📏 Tokens máximos")
                
                # 🔊 Checkbox para activar/desactivar Text-to-Speech
                tts_checkbox = gr.Checkbox(label="🗣 Activar TTS (Text-to-Speech)", value=False)
                file_upload = gr.File(label="📂 Subir documento .docx", type="filepath")

        sidebar_state = gr.State(False)

        toggle_sidebar_btn.click(
            fn=lambda state: (not state, gr.update(visible=not state)), 
            inputs=[sidebar_state], 
            outputs=[sidebar_state, sidebar]
        )

        voice_input_btn.click(speech_to_text, inputs=[], outputs=prompt_input, show_progress=True)
        file_upload.change(cargar_documento, inputs=[file_upload], outputs=chatbot)
        
        submit_btn.click(chat_with_ollama, 
                         inputs=[prompt_input, top_k_slider, top_p_slider, temperatura_slider, max_tokens_slider, tts_checkbox], 
                         outputs=chatbot)
    
    return demo

demo = crear_interfaz()
demo.launch(share=True)
