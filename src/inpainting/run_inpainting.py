import os
from PIL import PngImagePlugin
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
import torch
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)  # Increase the limit to 100 MB

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
SD_INPAINTING_DIR = os.path.join(PROJECT_ROOT, "foundation_models/sd-inpainting/")


def main(ch_object_image, ch_object_damage_mask):
    #model_id = "runwayml/stable-diffusion-inpainting"
    #model_id = "stabilityai/stable-diffusion-2-inpainting"
    #model_id = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"
    model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    '''pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")'''
    
    pipeline = AutoPipelineForInpainting.from_pretrained(SD_INPAINTING_DIR, 
                                                         torch_dtype=torch.float16
                                                         ).to("cuda")
    # Uncomment the following lines if you want to download the model from Hugging Face
    '''pipeline = AutoPipelineForInpainting.from_pretrained(model_id, 
                                                         torch_dtype=torch.float16, 
                                                         variant="fp16"
                                                         ).to("cuda")
    pipeline.save_pretrained(SD_INPAINTING_DIR)'''

    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_model_cpu_offload()

    prompt = "This same object, with consistent style, no damages, texture, and details."  # Fixed prompt

    init_image_xres, init_image_yres = ch_object_image.size
    inpainted_image = ch_object_image

    try:
        generator = torch.Generator(device="cuda").manual_seed(38)
        inpainted_image = pipeline(
            prompt=prompt,
            image=ch_object_image,
            mask_image=ch_object_damage_mask,
            generator=generator,
            strength=0.99,
            num_inference_steps=25,
        ).images[0].resize((init_image_xres, init_image_yres))
    except Exception as e:
        print(f"Error during generation: {str(e)}")

    print("Processing complete!")

    return inpainted_image