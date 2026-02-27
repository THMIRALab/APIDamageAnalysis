import torch
import torch.nn as nn
import torch.nn.functional as F


class Dinov2WithAdapters(nn.Module):
    def __init__(self, base_model="facebook/dinov2-base"):
        super(Dinov2WithAdapters, self).__init__()
        
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for param in self.dinov2.parameters():
            param.requires_grad = False  # Freeze backbone
        
        self.adapter = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # Project back to original dimension
            nn.Conv2d(256, 768, kernel_size=1),
            nn.BatchNorm2d(768)
        )
        
        self.classifier = ContextAwareClassifier(input_dim=768)

        self.classification_loss = nn.BCELoss()
        self.patch_size = 14

        self.segmentation_head = SegmentationHead()


    def forward(self, images, masks=None):
        # Extract DINOv2 features
        with torch.no_grad():  # No gradients for DINOv2
            features = self.dinov2.get_intermediate_layers(images)[0]
        
         # Get image dimensions and compute grid
        B, C, H, W = images.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        assert grid_h * grid_w == features.shape[1], \
            f"Grid {grid_h}x{grid_w} incompatible with seq_len {features.shape[1]}"

        # Reshape features to (B, hidden_size, grid_h, grid_w)
        spatial_features = features.view(B, grid_h, grid_w, -1).permute(0, 3, 1, 2)

        # Apply adapter and classifier
        adapted_features = self.adapter(spatial_features)  # (B, D, grid_h, grid_w)
        
        if masks is not None:
            return adapted_features
        else:
            patch_predictions = self.classifier(adapted_features).squeeze(1)
            return patch_predictions, adapted_features


    def masked_pooling(self, features, mask):
        B, D, H, W = features.shape
        mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")  # (B, 1, H, W)
        
        masked_features = features * mask  # (B, D, H, W)
        return masked_features.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)  # (B, D)
    

    def get_classifier_parameters(self):
        params = []
        params.extend(self.classifier.parameters())
        params.extend(self.adapter.parameters())
        return params
    

    def get_adapter_parameters(self):
        params = []
        params.extend(self.adapter.parameters())
        return params
    
    def get_segmentation_parameters(self):
        params = []
        params.extend(self.segmentation_head.parameters())
        params.extend(self.adapter.parameters())
        return params
    
    def get_sam_segmentation_parameters(self):
        params = []
        params.extend(self.unet_with_sam.parameters())
        params.extend(self.hiera_encoder.parameters())
        return params
    

class ContextAwareClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2  # To maintain spatial dimensions
        
        self.conv_net = nn.Sequential(
            # Convolution to aggregate neighboring patch features
            nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            
            # Final classification layer
            nn.Conv2d(hidden_dim, 1, kernel_size=1),  # 1x1 convolution
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input shape: (B, D, H, W)
        return self.conv_net(x)  # Output shape: (B, 1, H, W)
    

class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.refine = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )
        
        # Upsampling path (32x32 → 448x448)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # → 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # → 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),   # → 256x256
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),    # → 512x512
        )
    

    def forward(self, heatmap, object_mask):
        x = heatmap.unsqueeze(1)  # (B,1,32,32)
        x = self.refine(x)        # Refine details
        x = self.upsample(x)      # Upsample
        x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False) 

        return x * object_mask