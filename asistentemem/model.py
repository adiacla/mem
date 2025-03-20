import torch
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

enable_tts = False
# Global variables to store the model, tokenizer, and pipeline
model = None
tokenizer = None
pipe = None


def initialize_model():
    """Initialize the Llama 3.2 model with 4-bit quantization"""
    global model, tokenizer, pipe

    if pipe is not None:
        return  # Model already initialized

    # Login to Hugging Face (you may want to use environment variables for the token in production)
    login(token="")

    # Model ID
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    # Create 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load model with quantization config
    print("Loading Llama 3.2 model with 4-bit quantization...")

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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
    print("Model loaded successfully")


def chat_with_ollama(prompt, top_k, top_p, temperatura, max_tokens, tts_enabled):
    """Get response from Llama 3.2 model instead of Ollama"""
    global enable_tts, pipe
    enable_tts = tts_enabled

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

    # Create messages format for llama model
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

        assistant_message = outputs[0]["generated_text"][-1]["content"]

        # Generate audio if TTS is enabled
        audio_path = generate_tts_audio(assistant_message) if tts_enabled else None

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_message},
        ], audio_path
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message)
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": error_message},
        ], None


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

    print("Model resources cleaned up")
