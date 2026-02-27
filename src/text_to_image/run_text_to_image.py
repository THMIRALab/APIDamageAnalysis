import os
import pdfplumber
import whisper
import torch
import gc

from diffusers import DiffusionPipeline
from pathlib import Path
from mistral_inference.transformer import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage

MISTRAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "../foundation_models/mistral/7B-Instruct-v0.3/")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
SD_DIR = os.path.join(PROJECT_ROOT, "foundation_models/sd-xl/")


def extract_pdf_text(pdf_path):
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def transcribe_audio(audio_path, model_size="base"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]


def load_mistral():
    tokenizer = MistralTokenizer.from_file(
        f"{MISTRAL_DIR}/tokenizer.model.v3"
    )
    model = Transformer.from_folder(MISTRAL_DIR)
    return tokenizer, model


def load_sd():
    model = DiffusionPipeline.from_pretrained(
        SD_DIR,
        torch_dtype=torch.float16
    ).to("cuda")
    model.enable_xformers_memory_efficient_attention()
    
    # If downloading the model from Hugging Face, uncomment the following lines:
    '''model = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to("cuda")
    model.enable_xformers_memory_efficient_attention()
    model.save_pretrained(SD_DIR)'''
    
    return model


def generate_sd_prompt(tokenizer, model, story_text):
    system_msg = (
        "You are an assistant that extracts only the VISUAL appearance "
        "description of a cultural‑heritage object from a narrative and writes "
        "it as a concise Stable Diffusion prompt. Start directly with the "
        "description; do not add commentary. Summarize the text, but it must contain every characteristic of the object." )
    
    req = ChatCompletionRequest(
        messages=[
            SystemMessage(content=system_msg),
            UserMessage(content=story_text)
        ]
    )

    tokens = tokenizer.encode_chat_completion(req).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=68, temperature=0.3,
                             eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    return tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0]) + " The image must show the whole artifact."


def generate_image(pipe, prompt, negative_prompt=None):
    with torch.inference_mode():
        image = pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt,
            height=1024,
            width=1024,
            max_sequence_length=524
        ).images[0]
    #image.save(output_path)

    return image


def main(input_type, text=None, audio_path=None, document_path=None):
    if input_type == "text":
        story_text = text
        if story_text is None:
            raise ValueError("--text must be provided when input_type is 'text'.")
    elif input_type == "document":
        if document_path is None:
            raise ValueError("--document_path must be provided when input_type is 'document'.")
        if document_path.lower().endswith(".pdf"):
            story_text = extract_pdf_text(document_path)
        else:
            story_text = Path(document_path).read_text(encoding="utf-8")
    elif input_type == "audio":
        if audio_path is None:
            raise ValueError("--audio_path must be provided when input_type is 'audio'.")
        story_text = transcribe_audio(audio_path)
    else: 
        raise ValueError("Unsupported input_type.")
    
    print("[INFO] Loading Mistral‑7B‑Instruct…")
    tokenizer, model = load_mistral()
    print("[INFO] Generating image prompt…")
    sd_prompt = generate_sd_prompt(tokenizer, model, story_text)
    print("[INFO] Generated prompt:\n", sd_prompt)

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("[INFO] Loading Stable Diffusion pipeline…")
    sd_model = load_sd()
    print("[INFO] Generating image…")
    generated_image = generate_image(sd_model, sd_prompt)

    del sd_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return generated_image



def test_audio_to_image():
    """Test the text-to-image pipeline using an audio file as input."""
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the audio file (in the same directory)
    audio_path = os.path.join(current_dir, "Amulet2_AUDIO.mp3")
    
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"[INFO] Using audio file: {audio_path}")
    
    try:
        # Call the main function with audio input type
        generated_image = main(input_type="audio", audio_path=audio_path)
        
        # Save the generated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"generated_amulet_{timestamp}.png"
        output_path = os.path.join(current_dir, output_filename)
        generated_image.save(output_path)
        print(f"[SUCCESS] Image saved to: {output_path}")
        
        # Display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(generated_image)
        plt.axis('off')
        plt.title("Generated Image from Audio Description")
        plt.show()
        
        return generated_image
        
    except Exception as e:
        print(f"[ERROR] Failed to generate image: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the test when the script is executed directly
    test_audio_to_image()