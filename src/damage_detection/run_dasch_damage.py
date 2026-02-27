import os
import torch
import numpy as np
from .dasch_damage import Dinov2WithAdapters


def main(ch_object_image, ch_object_shape, model_name):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "models/"+model_name+"_DASCHNet.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Dinov2WithAdapters().to(device)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    with torch.no_grad():
        ch_object_image = ch_object_image.to(device)

        ch_object_shape_np = np.array(ch_object_shape)
        # Convert to tensor with proper dimensions [B,C,H,W]
        ch_object_shape_tensor = torch.from_numpy(ch_object_shape_np).float()
        if ch_object_shape_tensor.dim() == 2:  # If it's just [H,W]
            ch_object_shape_tensor = ch_object_shape_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif ch_object_shape_tensor.dim() == 3:  # If it's [C,H,W]
            ch_object_shape_tensor = ch_object_shape_tensor.unsqueeze(0)  # Add batch dim
        # Move to same device as model
        ch_object_shape_tensor = ch_object_shape_tensor.to(device)

        patch_preds, _ = model(ch_object_image)  # (B, H, W)
        damage_mask = model.segmentation_head(patch_preds, ch_object_shape_tensor)

        return damage_mask