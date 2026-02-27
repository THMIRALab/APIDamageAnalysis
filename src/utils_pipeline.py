import torch
import numpy as np

from src.classification.run_classification import main as object_classification
from shape_extraction.run_dasch_shape import main as get_shape_prior
from damage_detection.run_dasch_damage import main as get_damage_mask
from inpainting.run_inpainting import main as execute_damage_inpainting
from PIL import Image
from torchvision import transforms
from rembg import remove
from enum import Enum

class ObjectType(int, Enum):
    ARROWHEAD = 0
    STATUE = 1
    TABLET = 2
    VASE = 3

class InputType(str, Enum):
    TEXT = "text"
    DOCUMENT = "document"
    AUDIO = "audio"

MODEL_NAMES = {
    ObjectType.ARROWHEAD: "arrowhead",
    ObjectType.STATUE: "statue",
    ObjectType.TABLET: "tablet",
    ObjectType.VASE: "vase"
}


def extract_shape(ch_object_image, ch_object_image_mask, object_class):
    try:
        object_class = ObjectType(object_class)
    except ValueError:
        raise ValueError(f"Invalid object type: {object_class}")
    
    model_name = MODEL_NAMES[object_class]

    return get_shape_prior(ch_object_image, ch_object_image_mask, model_name)


def detect_damage(ch_object_image, ch_object_shape, object_class):
    try:
        object_class = ObjectType(object_class)
    except ValueError:
        raise ValueError(f"Invalid object type: {object_class}")
    
    model_name = MODEL_NAMES[object_class]
    
    return get_damage_mask(ch_object_image, ch_object_shape, model_name)


def convert_damage_mask_to_image(damage_mask):
    final_damage_mask = torch.sigmoid(damage_mask)
    final_damage_mask = (final_damage_mask > 0.5).float()
    final_damage_mask = final_damage_mask.cpu()
    final_damage_mask = final_damage_mask.numpy().squeeze()
    final_damage_mask = (final_damage_mask - final_damage_mask.min()) / (final_damage_mask.max() - final_damage_mask.min() + 1e-8)
    final_damage_mask = (final_damage_mask * 255).astype(np.uint8)

    return Image.fromarray((final_damage_mask))


def restore_ch_object(ch_object_image):   
    output_removal = remove(ch_object_image)
    ch_object_image = output_removal.convert("RGB")
    ch_object_image_mask = output_removal.split()[-1]  #Alpha channel 

    preprocess = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    ch_object_image_tensor = preprocess(ch_object_image).unsqueeze(0)

    class_idx = object_classification(ch_object_image_tensor)
    print(f"Class index: {class_idx}.")

    ch_object_shape = extract_shape(ch_object_image_tensor, ch_object_image_mask, class_idx)
    print(f"CH object shape extracted.")

    ch_object_damage_mask = detect_damage(ch_object_image_tensor, ch_object_shape, class_idx)
    print("Damage mask generated.")
    final_damage_mask = convert_damage_mask_to_image(ch_object_damage_mask)
    
    # Uncommen the following lines to perform inpainting. I commented only for testing purposes. They should not be commented.
    inpainted_object = execute_damage_inpainting(ch_object_image, final_damage_mask)
    print("Inpainting completed.")
    # inpainted_object = None

    return class_idx, ch_object_shape, final_damage_mask, inpainted_object, ch_object_image
