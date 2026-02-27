import math
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net    
    

    def get_dim(self):
        return self.dim


class AdapterDINO(nn.Module):
    def __init__(self, block, bottleneck_dim=32) -> None:
        super(AdapterDINO, self).__init__()
        self.block = block
        in_dim = self.block.attn.qkv.in_features
        self.learn = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, in_dim),
            nn.GELU()
        )

    def forward(self, input):
        adapter_output = self.learn(input)
        combined_input = input + adapter_output
        final_output = self.block(combined_input)        
        return final_output
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, feature, skip):
        feature = self.up(feature)
        feature = torch.cat((feature, skip), dim=1)
        feature = self.conv(feature)

        return feature
    

class DamageSegHead(nn.Module):
    def __init__(self, diff_in_ch=1, feat_in_ch=114, mid_ch=32):
        super().__init__()
        in_channels = diff_in_ch + feat_in_ch
        self.conv1 = nn.Conv2d(in_channels, mid_ch, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_ch, 1, kernel_size=3, padding=1)

    def forward(self, diff_map, feat):
        x = torch.cat([diff_map, feat], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x
    

def visualize_dino_attention(output, image, patch_size=14):
    """
    Visualize DINO attention maps on a given image.

    Args:
        model: DINOv2 model.
        image: Input image tensor (1, C, H, W).
        patch_size: Size of patches (default is 14 for ViT-B/14).

    Returns:
        None. Displays the attention map overlayed on the image.
    """
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    with torch.no_grad():
        # 256 tokens correspond to a 16x16 patch grid
        num_patches = output.size(1)  # 256
        grid_size = int(math.sqrt(num_patches))  # 16 for 16x16 grid

        # Reduce the embedding dimensions (768) to a single scalar per patch
        patch_attention = output.mean(dim=-1).reshape(1, grid_size, grid_size)  # Shape: (1, 16, 16)

        # Interpolate the patch attention map to match the input image size
        attention_map = F.interpolate(patch_attention.unsqueeze(1), size=(image.shape[2], image.shape[3]), mode="bilinear")
        attention_map = attention_map.squeeze().cpu().numpy()  # Convert to NumPy for visualization

    # Visualize the attention map overlayed on the original image
    image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC format for visualization
    '''plt.imshow(image_np)
    plt.imshow(attention_map, cmap="jet", alpha=0.5)
    plt.title("DINO Attention Map")
    plt.axis("off")
    plt.show()'''
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original image
    ax1.imshow(vutils.make_grid(image, normalize=True).permute(1, 2, 0).cpu())
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Plot attention map overlay
    ax2.imshow(image_np)
    ax2.imshow(attention_map, cmap="jet", alpha=0.5)
    ax2.set_title("DINO Attention Map")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_image_patches_and_mask(original_input, mask, patch_tokens_2d, grid_size):
    """
    Visualize the original image, mask, and patch embeddings side by side.

    Args:
        original_input (torch.Tensor): Shape (B, C, H, W), the original input images.
        mask (torch.Tensor): Shape (B, 1, H, W), the corresponding damage masks.
        patch_tokens_2d (torch.Tensor): Shape (B, GS, GS, D), the patch embeddings reshaped into a grid.
        grid_size (int): The number of patches along one dimension (e.g., 16 for 16x16 grid).
    """
    # Select the first sample from the batch
    sample_image = original_input[0].permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
    sample_mask = mask[0, 0].cpu().numpy()  # (1, H, W) -> (H, W)
    sample_patches = patch_tokens_2d[0].cpu().numpy()  # (GS, GS, D)

    # Image dimensions
    H, W, _ = sample_image.shape
    patch_h, patch_w = H // grid_size, W // grid_size

    # Create a reconstructed image from patches (mock visualization)
    reconstructed_image = np.zeros((H, W))
    for i in range(grid_size):
        for j in range(grid_size):
            patch_value = sample_patches[i, j].mean()  # Mock: Use mean of embedding as a pixel intensity
            reconstructed_image[
                i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w
            ] = patch_value

    # Normalize reconstructed image for visualization
    reconstructed_image = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())

    # Plot the original image, mask, and reconstructed patches side by side
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axs[0].imshow(sample_image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Mask
    axs[1].imshow(sample_mask, cmap="gray")
    axs[1].set_title("Mask")
    axs[1].axis("off")

    # Reconstructed patches
    axs[2].imshow(reconstructed_image, cmap="viridis")
    axs[2].set_title("Reconstructed Patches")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_patch_and_mask(full_image, full_mask, patch_image, mask_patch, row_start, row_end, col_start, 
                             col_end, damage_fraction, expanded_mask_patch):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Show full image with patch boundary
    img = full_image.squeeze().permute(1, 2, 0).cpu().numpy()
    axes[0,0].imshow(img)
    rect = plt.Rectangle((col_start, row_start), col_end-col_start, row_end-row_start, 
                        fill=False, color='red', linewidth=2)
    axes[0,0].add_patch(rect)
    axes[0,0].set_title('Full Image')
    axes[0,0].axis('off')
    
    # Show full mask with patch boundary
    mask = full_mask.squeeze().cpu().numpy()
    axes[0,1].imshow(mask, cmap='gray')
    rect = plt.Rectangle((col_start, row_start), col_end-col_start, row_end-row_start, 
                        fill=False, color='red', linewidth=2)
    axes[0,1].add_patch(rect)
    axes[0,1].set_title('Full Mask')
    axes[0,1].axis('off')
    
    # Show patch
    patch = patch_image.squeeze().permute(1, 2, 0).cpu().numpy()
    axes[1,0].imshow(patch)
    axes[1,0].set_title('Patch')
    axes[1,0].axis('off')
    
    # Show mask patch
    '''mask_p = mask_patch.squeeze().cpu().numpy()
    axes[1,1].imshow(mask_p, cmap='gray')
    axes[1,1].set_title(f'Mask Patch\nDamage Fraction: {damage_fraction:.3f}')
    axes[1,1].axis('off')'''
    expanded_mask_p = expanded_mask_patch.squeeze().cpu().numpy()
    axes[1, 1].imshow(expanded_mask_p, cmap='gray')
    axes[1, 1].set_title(f'Expanded Mask Patch\nDamage Fraction: {damage_fraction:.3f}')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_batch(images, masks):
    # Take up to num_samples from the batch
    n = images.shape[0]
    
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.5*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n):
        # Handle image: (C,H,W) -> (H,W,C)
        img = images[idx].cpu().permute(1,2,0).numpy()
        # Normalize to [0,1] range if needed
        img = (img - img.min()) / (img.max() - img.min())
        
        # Handle mask: Remove channel dim and squeeze
        msk = masks[idx].cpu().squeeze().numpy()
        
        # Plot
        axes[idx,0].imshow(img)
        axes[idx,0].set_title(f'Image {idx+1}')
        axes[idx,0].axis('off')
        
        axes[idx,1].imshow(msk, cmap='gray')
        axes[idx,1].set_title(f'Mask {idx+1}')
        axes[idx,1].axis('off')
    
    plt.tight_layout()
    plt.show()


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, upsampling_factor=4):
        """
        A lightweight segmentation head for DINO embeddings.
        
        Args:
            in_channels (int): Number of input channels (embedding dimensions).
            num_classes (int): Number of output classes (binary segmentation = 1).
            upsampling_factor (int): Factor to upsample the embeddings to image resolution.
        """
        super(SegmentationHead, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.output_conv(x)
        x = self.upsample(x)

        return x
    

def visualize_embeddings_2d(embeddings_2d, original_input, num_channels=4):
        """Visualize 2D embeddings as heatmaps"""
        B, D, H, W = embeddings_2d.shape
        
        # Select first few channels
        #channels_to_show = min(num_channels, 768)
        channels_to_show = 4
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        # Plot original image
        img = original_input[0].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        for i in range(channels_to_show):                
            # Get single channel
            embedding_channel = embeddings_2d[0, i].cpu().numpy()
            
            # Normalize channel
            embedding_channel = (embedding_channel - embedding_channel.min()) / \
                            (embedding_channel.max() - embedding_channel.min())
            
            # Plot heatmap
            im = axes[i+1].imshow(embedding_channel, cmap='viridis')
            axes[i+1].set_title(f'Channel {i}')
            axes[i+1].axis('off')
            plt.colorbar(im, ax=axes[i+1])
        
        plt.tight_layout()
        plt.show()


'''class UNetFinal(nn.Module):
    def __init__(self):
        super().__init_()

        self.final = nn.Sequential(
            nn.ConvTranspose2d(144, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64+144, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, hiera1, up_2_1):
        reconstruction = torch.cat([up_2_1, hiera1], dim=1)
        return self.final(reconstruction)'''
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    

class DomainAdapter(nn.Module):
    def __init__(self, adapter) -> None:
        super(DomainAdapter, self).__init__()
        self.adapter = adapter
        dim = adapter.get_dim()
        self.domain_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        domain_features = self.domain_learn(x)
        refined_features = x + domain_features
        return refined_features