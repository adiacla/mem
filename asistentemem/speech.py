import speech_recognition as sr
from gtts import gTTS
import re


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


def generate_tts_audio(text):
    """Generate TTS audio using gTTS and save as an MP3 file."""
    tts = gTTS(text=text, lang="es")
    audio_file = "response.mp3"
    tts.save(audio_file)
    return audio_file
