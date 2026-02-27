from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
import torch.nn.functional as F
from src.utils import Adapter, Up, RFB_modified
from skimage.morphology import binary_dilation, binary_erosion


class DaschPrior(nn.Module):
    def __init__(self, checkpoint_path=None, sam2_cfg="sam2.1_hiera_l.yaml"):
        super(DaschPrior, self).__init__()

        #region SAM2
        self.sam2_cfg = sam2_cfg
        if checkpoint_path:
            sam2 = build_sam2(sam2_cfg, checkpoint_path)
        else:
            sam2 = build_sam2(sam2_cfg)

        del sam2.sam_mask_decoder
        del sam2.sam_prompt_encoder
        del sam2.memory_encoder
        del sam2.memory_attention
        del sam2.mask_downsample
        del sam2.obj_ptr_tpos_proj
        del sam2.obj_ptr_proj
        del sam2.image_encoder.neck
        
        self.hiera_encoder = sam2.image_encoder.trunk
        for param in self.hiera_encoder.parameters():
            param.requires_grad = False

        adapters = []
        for block in self.hiera_encoder.blocks:
            adapters.append(Adapter(block))
        
        self.hiera_encoder.blocks = nn.Sequential(*adapters)
        #endregion

        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)

        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, original_input):
        hiera1, hiera2, hiera3, hiera4 = self.hiera_encoder(original_input)

        rfb1 = self.rfb1(hiera1)
        rfb2 = self.rfb2(hiera2)
        rfb3 = self.rfb3(hiera3)
        rfb4 = self.rfb4(hiera4)

        out_decoder1 = self.up1(rfb4, rfb3)
        out_side1 = F.interpolate(self.side1(out_decoder1), scale_factor=16, mode='bilinear')

        out_decoder2 = self.up2(out_decoder1, rfb2)
        out_side2 = F.interpolate(self.side2(out_decoder2), scale_factor=8, mode='bilinear')

        out_decoder3 = self.up3(out_decoder2, rfb1)
        head = F.interpolate(self.head(out_decoder3), scale_factor=4, mode='bilinear')

        return head, out_side1, out_side2
    

    def combine_masks(self, object_mask, damage_mask):
        object_mask_np = object_mask.cpu().numpy()
        damage_mask_np = damage_mask.cpu().numpy()

        object_binary = np.zeros_like(object_mask_np)
        object_binary[object_mask_np > 0.5] = 1

        damage_binary = np.zeros_like(damage_mask_np)
        damage_binary[damage_mask_np > 0.5] = 1

        #self.visualize_mask_combination(object_mask[0].cpu().numpy().squeeze(), damage_mask[0].cpu().numpy().squeeze())
        #self.visualize_mask_combination(object_binary[0].squeeze(), damage_binary[0].squeeze())

        combined_mask_np = np.logical_or(object_binary, damage_binary).astype(np.uint8)
        return torch.from_numpy(combined_mask_np).to('cuda')
    

    def combine_masks_torch(img_masks, completed_mask):
        # Keep everything in torch domain
        img_masks_binary = (img_masks > 0.5).float()
        completed_mask_binary = (completed_mask > 0.5).float()
        
        # Logical OR in torch: max(a, b) for binary masks
        combined = torch.max(img_masks_binary, completed_mask_binary)
        return combined
    

    def extract_missing_parts_mask(self, object_mask, damage_mask):
        object_mask_np = object_mask.cpu().numpy()
        damage_mask_np = damage_mask.cpu().numpy()

        object_binary = np.zeros_like(object_mask_np)  
        object_binary[object_mask_np > 0.5] = 1

        damage_binary = np.zeros_like(damage_mask_np)
        damage_binary[damage_mask_np > 0.5] = 1

        missing_parts_mask = np.zeros_like(damage_binary)
        missing_parts_mask[object_binary != 1] = damage_binary[object_binary != 1]
        missing_parts_mask[missing_parts_mask < 0.5] = 0
        return torch.from_numpy(missing_parts_mask).to('cuda')

    def extract_edge_map(self, batch_mask):
        if batch_mask.requires_grad:
            batch_mask_np = batch_mask.detach().cpu().numpy()
        else:
            batch_mask_np = batch_mask.cpu().numpy()

        batch_edge_map = np.zeros_like(batch_mask_np)

        for b in range(batch_mask_np.shape[0]):  # Batch dimension
            for c in range(batch_mask_np.shape[1]):  # Channel dimension
                mask = batch_mask_np[b, c]  # Extract single mask (H, W)

                # Perform morphological operations to compute edges
                dilated = binary_dilation(mask)
                eroded = binary_erosion(mask)
                edge_map = dilated ^ eroded  # XOR to get edges

                # Store the edge map in the corresponding position
                batch_edge_map[b, c] = edge_map.astype(float)

        return torch.from_numpy(batch_edge_map).to(batch_mask.device)

    def visualize_mask_combination(self, object_mask, damage_mask, save_path=None):
        plt.figure(figsize=(15, 5))
        
        # Object mask
        plt.subplot(1, 3, 1)
        plt.imshow(object_mask, cmap='grey')
        plt.title('Object Mask')
        plt.axis('off')
        
        # Damage mask
        plt.subplot(1, 3, 2)
        plt.imshow(damage_mask, cmap='grey')
        plt.title('Damage Mask')
        plt.axis('off')
        
        # Combined mask
        plt.subplot(1, 3, 3)
        combined_mask = np.logical_or(object_mask, damage_mask).astype(np.uint8)
        plt.imshow(combined_mask, cmap='grey')
        plt.title('Combined Mask')
        plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


    def visualize_combined_and_edge(self, image, combined_mask, edge_map):
        img_np = image[0].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        combined_np = combined_mask[0, 0].cpu().numpy()  # Extract first mask from batch
        edge_np = edge_map[0, 0].cpu().numpy()  # Extract first edge map from batch

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_np)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Combined Mask")
        plt.imshow(combined_np, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Edge Map")
        plt.imshow(edge_np, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


    # SAM2 generated some times very bad masks for the objects. The segmentation was not that good. That's why I am changing to another approach.
    def generate_object_mask_sam(self, image, mask):
        image_np = image[0].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    
        # 2. Denormalize from ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        
        # 3. Convert to uint8 range
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam2_image_predictor.set_image(image_np)
            all_sam_masks, _, _ = self.sam2_image_predictor.predict()
            combined_sam_mask = np.max(all_sam_masks, axis=0)  # Combine across mask dimension
            combined_sam_mask = (combined_sam_mask > 0.5).astype(np.uint8)

        
        #sam_mask = self.sam2_image_predictor.get_mask()
        self.visualize_sam_masks(image[0], combined_sam_mask, mask[0].cpu().numpy().squeeze())

        print(f"SAM mask shape: {combined_sam_mask.shape}, values: {np.unique(combined_sam_mask)}")
        print(f"GT mask shape: {mask[0].shape}, values: {np.unique(mask[0].cpu().numpy().squeeze())}")

        complete_shape_mask = np.logical_or(combined_sam_mask, mask[0].cpu().numpy().squeeze()).astype(np.uint8)
        self.visualize_sam_masks(image[0], complete_shape_mask, mask[0].cpu().numpy().squeeze())

        return complete_shape_mask, combined_sam_mask


    def visualize_sam_masks(self, image, sam_mask, original_mask, save_path=None):
        plt.figure(figsize=(12, 12))
        
        # Original image
        plt.subplot(2, 2, 1)
        img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        plt.imshow(img_np)
        plt.title('Original Image')
        plt.axis('off')
        
        # SAM mask
        plt.subplot(2, 2, 2)
        plt.imshow(sam_mask, cmap="grey")
        plt.title('SAM Mask')
        plt.axis('off')
        
        # Overlay
        plt.subplot(2, 2, 3)
        plt.imshow(img_np)
        plt.imshow(sam_mask, alpha=0.5, cmap='Reds')
        plt.title('Overlay')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(original_mask, cmap="grey")
        plt.title('Original Mask')
        plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()