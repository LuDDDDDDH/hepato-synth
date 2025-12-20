# -*- coding: utf-8 -*-
"""
multimodal_classifier.py

This script defines the multi-task diagnostic model for Study 3.

Core functionalities:
1. Defines a multi-task learning architecture based on the Swin Transformer backbone.
2. The model takes a fused, multi-channel tensor as input, which includes:
   - Original multi-phase MRI
   - Derived physical parameter maps
   - The AI-generated virtual Hepatobiliary Phase (vHBP) image
3. It simultaneously performs two tasks:
   a. Semantic Segmentation: To delineate the lesion boundaries.
   b. Multi-Class Classification: To diagnose the lesion type (HCC, FNH, etc.).
4. Leverages transfer learning from the encoder pre-trained in Study 1 to
   ensure a consistent and powerful feature representation.
"""

import logging
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from generative_models.physics_informed_unet import PhysicsInformedSwinUNETR

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLPHead(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) for classification."""
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout_rate: float = 0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.mlp(x)

class MultimodalDiagnosticSystem(nn.Module):
    """
    The integrated multi-task diagnostic system. It takes a fused multi-channel
    input and outputs both a segmentation mask and classification probabilities.
    """
    def __init__(self, model_config: dict):
        super().__init__()
        
        backbone_cfg = model_config['backbone']
        clf_head_cfg = model_config['classification_head']
        
        # --- 1. Instantiate the Backbone (Swin UNETR) ---
        # We start with the full PhysicsInformedSwinUNETR to easily handle
        # input layer modification and weight loading.
        logging.info("Initializing the Swin Transformer backbone for the diagnostic system...")
        self.backbone = PhysicsInformedSwinUNETR(
            img_size=tuple(backbone_cfg['img_size']),
            in_channels=backbone_cfg['in_channels'],
            out_channels=backbone_cfg['out_channels_seg'], # Set out_channels for segmentation
            feature_size=backbone_cfg['feature_size'],
            pretrained_weights_path=backbone_cfg['pretrained_weights_path']
        )
        
        # The segmentation head is already part of the SwinUNETR's decoder and output layer.
        # We can consider `self.backbone` as the "segmentation branch".
        
        # --- 2. Create the Classification Head ---
        logging.info("Initializing the classification head...")
        # We need to calculate the number of features at the bottleneck of the encoder.
        # For SwinUNETR, the bottleneck feature map is `hidden_states[-1]`.
        # Its size is (B, C, D/32, H/32, W/32). We'll use Global Average Pooling.
        # The number of channels C is feature_size * 16.
        bottleneck_features = backbone_cfg['feature_size'] * 16
        
        self.classification_pool = nn.AdaptiveAvgPool3d(1)
        self.classification_head = MLPHead(
            in_features=bottleneck_features,
            hidden_features=clf_head_cfg['hidden_features'],
            out_features=clf_head_cfg['out_features'],
            dropout_rate=clf_head_cfg.get('dropout_rate', 0.5)
        )
        
        logging.info("Multimodal diagnostic system initialized successfully.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-task learning.

        Args:
            x (torch.Tensor): The fused 7-channel input tensor (B, 7, D, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - seg_logits (torch.Tensor): The raw output for the segmentation task.
                                             Shape: (B, num_seg_classes, D, H, W).
                - clf_logits (torch.Tensor): The raw output for the classification task.
                                             Shape: (B, num_clf_classes).
        """
        # --- Pass through the shared backbone encoder ---
        # The swinViT is the encoder part of the SwinUNETR
        hidden_states = self.backbone.swin_unetr.swinViT(x)
        
        # --- Segmentation Path ---
        # The decoder takes the hidden states from the encoder
        seg_logits = self.backbone.swin_unetr.decoder(hidden_states)
        seg_logits = self.backbone.swin_unetr.out(seg_logits)
        
        # --- Classification Path ---
        # Take the deepest feature map from the encoder (the bottleneck)
        bottleneck_features = hidden_states[-1]
        
        # Apply global average pooling to get a single feature vector per sample
        pooled_features = self.classification_pool(bottleneck_features)
        
        # Flatten the features for the MLP
        flattened_features = torch.flatten(pooled_features, 1)
        
        # Get classification logits
        clf_logits = self.classification_head(flattened_features)
        
        return seg_logits, clf_logits


if __name__ == '__main__':
    # --- Example Usage & Sanity Check ---
    
    # These parameters would come from study_3_diagnostic.yaml
    config = {
        'backbone': {
            'img_size': (96, 96, 96),
            'in_channels': 7,
            'out_channels_seg': 6, # 5 lesion classes + 1 background
            'feature_size': 48,
            'pretrained_weights_path': None
        },
        'classification_head': {
            'hidden_features': 512,
            'out_features': 5, # 5 lesion classes
            'dropout_rate': 0.5
        }
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the model
    model = MultimodalDiagnosticSystem(config).to(device)
    
    # 2. Print a part of the model to check
    print("Backbone Input Layer:", model.backbone.swin_unetr.swinViT.patch_embed.proj)
    print("\nClassification Head:", model.classification_head)
    
    # 3. Create a dummy input tensor
    dummy_input = torch.randn(2, config['backbone']['in_channels'], *config['backbone']['img_size']).to(device)
    print(f"\nInput tensor shape: {dummy_input.shape}")
    
    # 4. Run a forward pass
    model.train() # Set to train mode for dropout to be active
    seg_output, clf_output = model(dummy_input)
    
    # 5. Check if the output shapes are correct
    print(f"Segmentation output shape: {seg_output.shape}")
    assert seg_output.shape == (2, config['backbone']['out_channels_seg'], *config['backbone']['img_size'])
    
    print(f"Classification output shape: {clf_output.shape}")
    assert clf_output.shape == (2, config['classification_head']['out_features'])
    
    print("\nMultimodal diagnostic system instantiation and forward pass successful!")