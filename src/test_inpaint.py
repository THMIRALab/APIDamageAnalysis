import gc
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
from matplotlib import pyplot as plt
import torch
from PIL import Image, PngImagePlugin
from diffusers.utils import load_image
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_pipeline import InputType, restore_ch_object

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)  # Increase the limit to 100 MB

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SD_INPAINTING_DIR = os.path.join(PROJECT_ROOT, "foundation_models/sd-inpainting/")


def executeSDInpaintingPipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    #model_id = "runwayml/stable-diffusion-inpainting"
    #model_id = "stabilityai/stable-diffusion-2-inpainting"
    #model_id = "stabilityai/stable-diffusion-xl-base-1.0"
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
                                                         ).to("cuda")'''
    pipeline.enable_model_cpu_offload()

    for i in range(0, 5):
        '''initial_image_file_name = input("Provide the name of the image.")
        mask_file_name = input("Provide the name of the mask file.")
        prompt = input("Provide the prompt.")'''

        path_value_test = "test_fixed/statue/"
        path_mask_value_test = "test_fixed/statue/"

        initial_image_file_name = "statue_damaged_item_G5ISXTGZBMBQK7X7L7RFYT6XC7CEPFVB.png"
        mask_file_name = "generated_mask_statue_damaged_item_G5ISXTGZBMBQK7X7L7RFYT6XC7CEPFVB.png"
        prompt = "This same object, with consistent style, no damages, texture, and details."
        #prompt = "This arrowhead, realistically restored. Preserve the original object, existing color, texture, material, and style."  # Fixed prompt
        #prompt = "Make the right and left hands of the statue. Preserve the existing color, texture, material, and style."

        init_image = load_image(os.path.join(current_dir, path_value_test, initial_image_file_name))
        init_image_xres, init_image_yres = init_image.size
        mask_image = load_image(os.path.join(current_dir, path_mask_value_test, mask_file_name))

        print("Starting inpainting...")
        start_generation = time.time()

        generator = torch.Generator(device="cuda").manual_seed(0)
        try:
            inpainted_image = pipeline(
                prompt=prompt,
                image=init_image,
                mask_image=mask_image,
                generator=generator,
                strength=0.99,
                num_inference_steps=25,
            ).images[0].resize((init_image_xres, init_image_yres))
        except Exception as e:
            print(f"Error during generation: {str(e)}")

        #inpainted_image_name = "./SDInpaintingPipeline-" + initial_image_file_name
        inpainted_image_name = initial_image_file_name
        inpainted_image.save(inpainted_image_name)

        end_generation = time.time()
        print("The inpainting process took: " + str(round(end_generation - start_generation, 2)) + " seconds.")

        fig = plt.figure(figsize=(15, 10))
        fig.add_subplot(1, 2, 1)
        plt.imshow(init_image)
        plt.axis("off")
        plt.title("Original image")

        fig.add_subplot(1, 2, 2)
        plt.imshow(inpainted_image)
        plt.axis("off")
        plt.title("Inpainted image")

        plt.show()


def execute_complete_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    #model_id = "runwayml/stable-diffusion-inpainting"
    #model_id = "stabilityai/stable-diffusion-2-inpainting"
    #model_id = "stabilityai/stable-diffusion-xl-base-1.0"
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
                                                         ).to("cuda")'''
    pipeline.enable_model_cpu_offload()

    '''image_dir = os.path.join(current_dir, "test_fixed/ValidationDataHeritageMalta/images")
    output_dir = os.path.join(current_dir, "test_fixed/ValidationDataHeritageMalta/results")

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]'''
    image_files = []

    for i, image_filename in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_filename}")

        try:
            init_image = load_image(os.path.join(image_dir, image_filename))
            init_image_xres, init_image_yres = init_image.size
            _, object_shape, mask_image, _, image_no_background = restore_ch_object(init_image)
            
            prompt = "This same object, with consistent style, no damages, texture, and details."
            
            print(f"Starting inpainting for {image_filename}...")
            start_generation = time.time()
            
            generator = torch.Generator(device="cuda").manual_seed(0)
            inpainted_image = pipeline(
                prompt=prompt,
                image=image_no_background,
                mask_image=mask_image,
                generator=generator,
                strength=0.99,
                num_inference_steps=25,
            ).images[0].resize((init_image_xres, init_image_yres))
            
            # Save the result
            output_path = os.path.join(output_dir, f"inpainted_{image_filename}")
            inpainted_image.save(output_path)
            
            end_generation = time.time()
            print(f"Inpainting for {image_filename} completed in {round(end_generation - start_generation, 2)} seconds.")
            
            # Display results
            fig = plt.figure(figsize=(20, 15))  # Larger figure size for 4 images

            # Original image (top left)
            fig.add_subplot(2, 2, 1)
            plt.imshow(init_image)
            plt.axis("off")
            plt.title("Original image")

            # Object shape (top right)
            fig.add_subplot(2, 2, 2)
            plt.imshow(object_shape, cmap='gray')  # Using a colormap appropriate for mask data
            plt.axis("off")
            plt.title("Object shape")

            # Damage mask (bottom left)
            fig.add_subplot(2, 2, 3)
            plt.imshow(mask_image, cmap='gray')  # Masks are typically better shown in grayscale
            plt.axis("off")
            plt.title("Damage mask")

            # Inpainted result (bottom right)
            fig.add_subplot(2, 2, 4)
            plt.imshow(inpainted_image)
            plt.axis("off")
            plt.title("Inpainted result")

            plt.tight_layout()  # Adjust spacing between subplots
            plt.show()
            
        except Exception as e:
            print(f"Error processing {image_filename}: {str(e)}")



if __name__ == "__main__":
    executeSDInpaintingPipeline()
    #execute_complete_pipeline()