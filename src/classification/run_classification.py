import os
import torch
import sys
from .dasch_classification import DASCHClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HIERA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "../sam2/checkpoints/sam2.1_hiera_large.pt")
HIERA_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "models/classification_DASCHNet.pth")


def main(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DASCHClassification(HIERA_PATH, HIERA_CFG).to(device)
    model.load_state_dict(torch.load(MODEL_PATH), strict=True)
    model.eval()
    
    with torch.no_grad():
        image = image.to(device)
            
        classification_output = model(image)
        _, prediction = torch.max(classification_output, 1)
    
    print(f"Classification Prediction: {model.class_names[prediction.item()]}")

    return prediction.item()