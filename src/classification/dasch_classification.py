import torch.nn as nn
from sam2.build_sam import build_sam2
from src.utils import Adapter, Up, RFB_modified


class DASCHClassification(nn.Module):
    def __init__(self, checkpoint_path=None, sam2_cfg="sam2.1_hiera_l.yaml", num_classes=4):
        super(DASCHClassification, self).__init__()

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
        
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        
        # Replace final conv with classification layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        self.class_to_idx = {
            "arrowhead": 0,
            "statue": 1,
            "tablet": 2,
            "vase": 3
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.class_names = ["arrowhead", "statue", "tablet", "vase"]


    def forward(self, x):
        hiera1, hiera2, hiera3, hiera4 = self.hiera_encoder(x)
        
        # Apply RFB modules
        rfb1 = self.rfb1(hiera1)
        rfb2 = self.rfb2(hiera2)
        rfb3 = self.rfb3(hiera3)
        rfb4 = self.rfb4(hiera4)
        
        # Use the decoder path to fuse multi-scale features
        out_decoder1 = self.up1(rfb4, rfb3)
        out_decoder2 = self.up2(out_decoder1, rfb2)
        out_decoder3 = self.up3(out_decoder2, rfb1)
        
        # Global pooling and classification
        features = self.global_pool(out_decoder3).flatten(1)
        logits = self.classifier(features)
        
        return logits