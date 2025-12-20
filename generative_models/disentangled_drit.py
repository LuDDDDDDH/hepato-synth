# -*- coding: utf-8 -*-
"""
disentangled_drit.py

This script defines the model architecture for Study 2: the Disentangled
Representation for Image-to-Image Translation (DRIT) model, adapted with a
Swin Transformer backbone (Swin-DRIT++).

This model is designed for unpaired image-to-image translation, learning to
map from a source domain (e.g., standard contrast MRI) to a target domain
(e.g., virtual HBP) without paired examples.

The architecture consists of:
- A shared Content Encoder (reusing the pre-trained Swin UNETR encoder).
- Domain-specific Style Encoders.
- A shared Generator/Decoder.
- Domain-specific PatchGAN Discriminators.
"""

import logging
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from .physics_informed_unet import PhysicsInformedSwinUNETR # To reuse the encoder

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# Helper Modules
# ==============================================================================

class LightweightStyleEncoder(nn.Module):
    """
    A lightweight CNN to encode an image into a low-dimensional style vector.
    It uses a series of down-sampling convolutional layers followed by global
    average pooling.
    """
    def __init__(self, in_channels: int, style_dim: int, num_layers: int = 4):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        mult = 1
        for i in range(1, num_layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            layers += [
                nn.Conv3d(64 * mult_prev, 64 * mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm3d(64 * mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            
        layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(nn.Conv3d(64 * mult, style_dim, kernel_size=1, stride=1, padding=0))
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PatchGANDiscriminator(nn.Module):
    """
    A 3D PatchGAN discriminator that classifies if patches in an image are real or fake.
    """
    def __init__(self, in_channels: int, num_layers: int = 4):
        super().__init__()
        
        layers = [
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        mult = 1
        for i in range(1, num_layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            layers += [
                nn.Conv3d(64 * mult_prev, 64 * mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm3d(64 * mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers.append(nn.Conv3d(64 * mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.discriminator = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


# ==============================================================================
# Main DisentangledDRIT Model
# ==============================================================================

class DisentangledDRIT(nn.Module):
    """
    The main Swin-DRIT++ model, orchestrating all sub-networks.
    """
    def __init__(self, model_config: dict):
        super().__init__()
        
        # --- 1. Instantiate the Content Encoder ---
        # We load the full Study 1 model and then extract its encoder part.
        logging.info("Initializing Content Encoder from pre-trained Study 1 model...")
        study1_model_cfg = model_config['content_encoder']
        full_study1_model = PhysicsInformedSwinUNETR(
            img_size=tuple(study1_model_cfg['img_size']),
            in_channels=study1_model_cfg['in_channels'],
            out_channels=1, # Dummy out_channels
            feature_size=study1_model_cfg['feature_size'],
            pretrained_weights_path=study1_model_cfg['pretrained_weights_path']
        )
        self.E_content = full_study1_model.swin_unetr.swinViT
        
        # Freeze the content encoder as per the config
        if study1_model_cfg.get('freeze_weights', False):
            logging.info("Freezing weights of the Content Encoder.")
            for param in self.E_content.parameters():
                param.requires_grad = False
        
        # --- 2. Instantiate Style Encoders for each domain ---
        style_cfg = model_config['style_encoder']
        # Domain A: Standard Contrast MRI (multi-channel input)
        self.E_style_A = LightweightStyleEncoder(in_channels=study1_model_cfg['in_channels'], style_dim=style_cfg['style_dim'])
        # Domain B: Hepatobiliary Phase (single-channel input)
        self.E_style_B = LightweightStyleEncoder(in_channels=1, style_dim=style_cfg['style_dim'])

        # --- 3. Instantiate the Generator (Decoder) ---
        # The generator will combine content and style features. This is a simplification.
        # A true DRIT model has a more complex generator. Here we reuse the Swin UNETR decoder.
        # This requires careful handling of feature fusion (e.g., AdaIN).
        # For simplicity, we'll represent it with the decoder part of SwinUNETR.
        # The fusion logic will be in the forward pass.
        self.G_decoder = full_study1_model.swin_unetr.decoder
        self.G_out = full_study1_model.swin_unetr.out

        # The fusion of style and content is complex. A simple approach is to have MLP
        # layers that transform the style code and add it to the content features.
        # This is a placeholder for that logic.
        
        # --- 4. Instantiate Discriminators for each domain ---
        disc_cfg = model_config['discriminator']
        # Discriminator for Domain A (judges standard contrast images)
        self.D_A = PatchGANDiscriminator(in_channels=study1_model_cfg['in_channels'], num_layers=disc_cfg['num_layers'])
        # Discriminator for Domain B (judges HBP images)
        self.D_B = PatchGANDiscriminator(in_channels=1, num_layers=disc_cfg['num_layers'])
        
        logging.info("Swin-DRIT++ model initialized successfully.")

    def forward(self, x_A: torch.Tensor, x_B: torch.Tensor):
        """
        A simplified forward pass illustrating the core translation logic.
        The full training loop will call these components separately.

        Args:
            x_A (torch.Tensor): An image from Domain A (standard contrast).
            x_B (torch.Tensor): An image from Domain B (HBP).
        
        Returns:
            A dictionary of generated and reconstructed images.
        """
        # --- Self-reconstruction path ---
        # Reconstruct x_A using its own content and style
        content_A = self.E_content(x_A)[-1] # Get the last feature map from the encoder
        style_A = self.E_style_A(x_A)
        # x_A_recon = self.G(content_A, style_A) # Simplified
        
        # Reconstruct x_B using its own content and style
        # Note: The content encoder is trained on Domain A. Applying it to Domain B is part of the assumption.
        content_B = self.E_content(self.convert_to_encoder_input(x_B))[-1]
        style_B = self.E_style_B(x_B)
        # x_B_recon = self.G(content_B, style_B) # Simplified

        # --- Cross-domain translation path ---
        # Translate x_A to Domain B
        # x_A_to_B = self.G(content_A, style_B)
        
        # Translate x_B to Domain A
        # x_B_to_A = self.G(content_B, style_A)

        # --- Cycle-reconstruction path ---
        # Reconstruct x_A from the translated image
        # content_A_from_B = self.E_content(x_A_to_B) # This is complex...
        # style_B_from_A = self.E_style_B(x_A_to_B)
        # x_A_cycle = self.G(self.E_content(x_B_to_A)[-1], style_A)

        # The actual implementation of the forward pass is highly complex and
        # is usually handled by the training script which orchestrates
        # the calls to each sub-network. This forward function is conceptual.
        pass

    def convert_to_encoder_input(self, x_b: torch.Tensor) -> torch.Tensor:
        """
        Helper to create a fake multi-channel input for the content encoder 
        when the input is from single-channel domain B.
        A simple strategy is to replicate the channel.
        """
        return x_b.repeat(1, self.E_content.in_channels, 1, 1, 1)

if __name__ == '__main__':
    # --- Example Usage & Sanity Check ---
    # This is a conceptual check as the full model is orchestrated by the trainer.
    
    # These would come from study_2_virtual.yaml
    config = {
        'content_encoder': {
            'img_size': (96, 96, 96), 'in_channels': 6, 'feature_size': 48,
            'pretrained_weights_path': None, 'freeze_weights': True
        },
        'style_encoder': {'style_dim': 8},
        'discriminator': {'num_layers': 4}
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the full model
    model = DisentangledDRIT(config).to(device)

    # 2. Check if all components are initialized
    print("Content Encoder:", model.E_content)
    print("\nStyle Encoder A:", model.E_style_A)
    print("\nDiscriminator B:", model.D_B)
    
    # 3. Create dummy inputs from both domains
    dummy_A = torch.randn(1, 6, 96, 96, 96).to(device)
    dummy_B = torch.randn(1, 1, 96, 96, 96).to(device)
    
    # 4. Check outputs of individual components
    with torch.no_grad():
        style_code_A = model.E_style_A(dummy_A)
        style_code_B = model.E_style_B(dummy_B)
        disc_out_B = model.D_B(dummy_B)

    print(f"\nInput shape A: {dummy_A.shape}")
    print(f"Output Style Code A shape: {style_code_A.shape}")
    assert style_code_A.shape == (1, 8, 1, 1, 1)
    
    print(f"\nInput shape B: {dummy_B.shape}")
    print(f"Output Style Code B shape: {style_code_B.shape}")
    assert style_code_B.shape == (1, 8, 1, 1, 1)

    print(f"\nDiscriminator B output shape: {disc_out_B.shape}")
    # The output size depends on the input size and number of layers in PatchGAN
    
    print("\nSwin-DRIT++ model components instantiated successfully!")