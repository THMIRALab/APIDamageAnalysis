import os
import torch
import numpy as np
import torch.nn.functional as F
from .dasch_shape import DaschPrior
from scipy import ndimage
from torchvision import transforms

HIERA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "../sam2/checkpoints/sam2.1_hiera_large.pt")
HIERA_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def combine_masks_torch(img_masks, completed_mask):
    completed_mask_prob = torch.sigmoid(completed_mask)
    
    img_masks_binary = (img_masks > 0.4).float()
    completed_mask_binary = (completed_mask_prob > 0.4).float()
    
    # Combine masks
    combined = torch.max(img_masks_binary, completed_mask_binary)
    
    # Apply morphological closing to smooth borders
    kernel_size = 5
    padding = kernel_size // 2
    combined = F.max_pool2d(combined, kernel_size=kernel_size, stride=1, padding=padding)
    
    return combined


def main(ch_object_image, ch_object_image_mask, model_name):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "models/"+model_name+"_DASCHNet.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DaschPrior(HIERA_PATH, HIERA_CFG).to(device)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    with torch.no_grad():
        ch_object_image = ch_object_image.to(device)

        preprocess = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor()
        ])
        ch_object_image_mask = preprocess(ch_object_image_mask).unsqueeze(0)
        ch_object_image_mask = ch_object_image_mask.to(device)
       
        completed_mask, _, _ = model(ch_object_image)

        complete_img_masks = combine_masks_torch(ch_object_image_mask, completed_mask)
        
        complete_img_masks = complete_img_masks.cpu().numpy()
        complete_img_masks = np.squeeze(complete_img_masks, axis=0)  # Now (1,H,W)
        complete_img_masks = np.squeeze(complete_img_masks)  # Now (H,W)

        # Apply morphological operations to smooth borders
        complete_img_masks = ndimage.binary_closing(complete_img_masks, structure=np.ones((3,3)))
        complete_img_masks = ndimage.binary_fill_holes(complete_img_masks)

        return (complete_img_masks * 255).astype(np.uint8)

        