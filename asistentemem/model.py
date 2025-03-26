import torch
import requests  # added for Ollama API call
import json
from transformers import (
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from huggingface_hub import login
import gc
from asistentemem.speech import generate_tts_audio
from asistentemem.data import get_document_text, get_api_text

# Global variable for API selection: if True, use Ollama API; if False, use Hugging Face transformers.
USE_OLLAMA_API = False

# Global variables to store the model, tokenizer, and pipeline
model = None
tokenizer = None
pipe = None


def initialize_model():
    """Initialize the Llama 3.2 model with 4-bit quantization"""
    global model, tokenizer, pipe

    if pipe is not None:
        return  # Model already initialized

    # print("[DEBUG] Logging into Hugging Face")
    # login(token="")  # Add your token here

    # Model ID
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    # Create 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("[DEBUG] Loading model with quantization config...")
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] Using device: {device}")

    # Load model and tokenizer separately with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create pipeline with loaded model and tokenizer
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    print("[DEBUG] Model loaded successfully")


def chat_with_huggingface(prompt, top_k, top_p, temperatura, max_tokens, tts_enabled):
    """Get response using Hugging Face transformers pipeline"""
    global pipe

    print(f"[DEBUG] Chatting with Hugging Face: prompt='{prompt}'")
    if not prompt.strip():
        return [
            {
                "role": "assistant",
                "content": "Documento subido correctamente. ¿Qué desea preguntar?",
            }
        ], None

    # Initialize model if not already loaded
    if pipe is None:
        initialize_model()

    document_text = get_document_text()
    api_text = get_api_text()
    print(
        f"[DEBUG] Document text length: {len(document_text)}; API text length: {len(api_text)}"
    )

    # Create messages format for the model
    messages = []

    # Add context if available
    if document_text or api_text:
        context = f"{document_text}\n\n{api_text}"
        messages.append(
            {"role": "system", "content": f"Context information: {context}"}
        )

    # Add user message
    messages.append({"role": "user", "content": prompt})

    try:
        # Generate response from the model
        outputs = pipe(
            messages,
            max_new_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperatura,
            do_sample=True,
        )
        print(f"[DEBUG] Raw model output: {outputs}")

        assistant_message = outputs[0]["generated_text"][-1]["content"]

        # Generate audio if TTS is enabled
        audio_path = generate_tts_audio(assistant_message) if tts_enabled else None

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_message},
        ], audio_path
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(f"[ERROR] {error_message}")
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": error_message},
        ], None


def chat_with_ollama_api(prompt, top_k, top_p, temperatura, max_tokens, tts_enabled):
    """Get response using the Ollama API call"""
    api_url = "http://localhost:11434/api/chat"

    document_text = get_document_text()
    api_text = get_api_text()

    context = ""
    if document_text or api_text:
        context = f"{document_text}\n\n{api_text}"

    payload = {
        "model": "llama3.2:3b-instruct-q6_K",
        "messages": [],
        "options": {
            "temperature": temperatura,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    if context:
        payload["messages"].append(
            {"role": "system", "content": f"Context information: {context}"}
        )

    payload["messages"].append({"role": "user", "content": prompt})

    print(f"[DEBUG] Sending payload to Ollama API: {json.dumps(payload, indent=2)}")
    try:
        response = requests.post(api_url, json=payload)
        print(f"[DEBUG] Ollama API response status: {response.status_code}")
        if response.ok:
            try:
                result = response.json()
            except ValueError as json_err:
                result = json.loads(response.text.splitlines()[0])
            print(f"[DEBUG] Raw Ollama API result: {json.dumps(result, indent=2)}")
            assistant_message = result.get("message", {}).get(
                "content", "No response content in API result"
            )
            print(
                f"[DEBUG] Extracted assistant message: '{assistant_message}' (length: {len(assistant_message)})"
            )
        else:
            assistant_message = f"❌ Error en API Ollama: {response.status_code}"
            print(f"[ERROR] {assistant_message}")
    except Exception as e:
        assistant_message = f"❌ Error al conectar con API Ollama: {str(e)}"
        print(f"[ERROR] {assistant_message}")

    audio_path = generate_tts_audio(assistant_message) if tts_enabled else None
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_message},
    ], audio_path


def chat(prompt, top_k, top_p, temperatura, max_tokens, tts_enabled):
    """Selects the API call based on the global preference"""
    print(f"[DEBUG] chat() called with prompt: {prompt}")
    if USE_OLLAMA_API:
        return chat_with_ollama_api(
            prompt, top_k, top_p, temperatura, max_tokens, tts_enabled
        )
    else:
        return chat_with_huggingface(
            prompt, top_k, top_p, temperatura, max_tokens, tts_enabled
        )


def cleanup_model():
    """Clean up the model resources to free memory"""
    global model, tokenizer, pipe

    if pipe is not None:
        del pipe
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer

    pipe, model, tokenizer = None, None, None
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[DEBUG] Model resources cleaned up")
