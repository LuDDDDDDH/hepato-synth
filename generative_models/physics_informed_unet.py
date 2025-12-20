# -*- coding: utf-8 -*-
"""
physics_informed_unet.py

This script defines the model architecture for Study 1: the Physics-Informed 
Swin UNETR.

Core functionalities:
1. Wraps the powerful SwinUNETR from the MONAI library.
2. Modifies the input layer of the standard SwinUNETR to accept a multi-channel
   input tensor, which includes both MRI sequences and physical parameter maps.
3. Implements a "weight surgery" mechanism to intelligently load pre-trained
   weights from a single-channel model onto the new multi-channel input layer.
4. Provides a clean, configurable model class that can be easily instantiated
   from the YAML configuration files.
"""

import logging
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PhysicsInformedSwinUNETR(nn.Module):
    """
    A custom wrapper for MONAI's SwinUNETR to support multi-channel inputs
    (MRI phases + physical maps) and facilitate transfer learning.
    """
    def __init__(self,
                 img_size: tuple,
                 in_channels: int,
                 out_channels: int,
                 feature_size: int = 48,
                 depths: tuple = (2, 2, 2, 2),
                 num_heads: tuple = (3, 6, 12, 24),
                 use_checkpoint: bool = False,
                 pretrained_weights_path: str = None):
        """
        Initializes the Physics-Informed Swin UNETR model.

        Args:
            img_size (tuple): The spatial size of the input patches (e.g., (96, 96, 96)).
            in_channels (int): Number of input channels (e.g., 6 for 4 MRI + 2 physics maps).
            out_channels (int): Number of output channels (e.g., 1 for the HBP image).
            feature_size (int): The base feature size for the Swin Transformer blocks.
            depths (tuple): The number of layers in each Swin Transformer stage.
            num_heads (tuple): The number of attention heads in each Swin Transformer stage.
            use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
            pretrained_weights_path (str, optional): Path to the pre-trained model weights
                                                     (typically from a model trained on a single channel).
                                                     If None, weights are initialized randomly.
        """
        super().__init__()
        
        # 1. Instantiate the base SwinUNETR model from MONAI
        # We initialize it with a placeholder in_channels=1, as we will replace
        # the input layer shortly.
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=1, # Placeholder
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint
        )

        # 2. Perform "Weight Surgery" on the input layer
        self.modify_input_layer(new_in_channels=in_channels, 
                                pretrained_weights_path=pretrained_weights_path)

    def modify_input_layer(self, new_in_channels: int, pretrained_weights_path: str):
        """
        Replaces the first convolutional layer to accept the desired number of
        input channels and intelligently loads pre-trained weights.
        """
        original_input_layer = self.swin_unetr.swinViT.patch_embed.proj
        original_weights = None
        
        # --- Load pre-trained weights if provided ---
        if pretrained_weights_path:
            logging.info(f"Loading pre-trained weights from: {pretrained_weights_path}")
            try:
                # Load the state dict from the .pth file
                pretrained_state_dict = torch.load(pretrained_weights_path, map_location='cpu')
                
                # If the state_dict is nested (common in MONAI checkpoints), find the model weights
                if 'model' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['model']
                elif 'state_dict' in pretrained_state_dict:
                    pretrained_state_dict = pretrained_state_dict['state_dict']

                # Load the weights into the entire model
                self.swin_unetr.load_state_dict(pretrained_state_dict, strict=False)
                
                # Extract the weights of the original input layer for our surgery
                original_weights = self.swin_unetr.swinViT.patch_embed.proj.weight
                logging.info("Successfully loaded pre-trained weights into the base model.")
            except Exception as e:
                logging.error(f"Failed to load pre-trained weights: {e}. Model will be randomly initialized.")
                pretrained_weights_path = None # Reset path to prevent using weights
        else:
            logging.info("No pre-trained weights provided. Model will be randomly initialized.")

        # --- Create the new input layer ---
        new_input_layer = nn.Conv3d(
            in_channels=new_in_channels,
            out_channels=original_input_layer.out_channels,
            kernel_size=original_input_layer.kernel_size,
            stride=original_input_layer.stride,
            padding=original_input_layer.padding
        ).to(original_input_layer.weight.device)
        
        # --- Intelligently transfer weights to the new layer ---
        if original_weights is not None:
            logging.info(f"Performing weight surgery on the input layer to adapt from 1 to {new_in_channels} channels.")
            with torch.no_grad():
                # The original weights have shape (out_channels, 1, k, k, k)
                # The new weights need shape (out_channels, new_in_channels, k, k, k)
                
                # Strategy:
                # - The first channel of the new layer gets the original pre-trained weights.
                # - The remaining channels are initialized differently. For MRI channels,
                #   we can use the mean of the original weights. For physics maps,
                #   which have a different data distribution, zero-initialization is safer.
                
                # Initialize new weights tensor
                new_weights = torch.zeros_like(new_input_layer.weight)
                
                # Copy original weights to the first channel
                new_weights[:, 0:1, :, :, :] = original_weights
                
                # Let's assume the first 4 channels are MRI, and last 2 are physics maps
                num_mri_channels = 4 # Based on our config
                if new_in_channels > 1:
                    # For other MRI channels (1 to 3), copy the mean of the original weights
                    mean_weights = torch.mean(original_weights, dim=1, keepdim=True)
                    for i in range(1, min(new_in_channels, num_mri_channels)):
                        new_weights[:, i:i+1, :, :, :] = mean_weights

                # For physics map channels (4 and 5), we leave them as zeros,
                # as their data distribution is very different from MRI signals.
                # This forces the model to learn their features from scratch without
                # harmful bias from the pre-trained weights.
                
                new_input_layer.weight.copy_(new_weights)
                
                # Copy the bias if it exists
                if original_input_layer.bias is not None:
                    new_input_layer.bias.copy_(original_input_layer.bias)
            logging.info("Weight surgery complete.")
        else:
            logging.info("New input layer is randomly initialized.")

        # --- Replace the old layer with the new one ---
        self.swin_unetr.swinViT.patch_embed.proj = new_input_layer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): The multi-channel input tensor of shape (B, C, D, H, W).
        
        Returns:
            torch.Tensor: The single-channel output tensor (generated HBP).
        """
        return self.swin_unetr(x)

if __name__ == '__main__':
    # --- Example Usage & Sanity Check ---
    
    # These parameters would come from the config file (e.g., study_1_acceleration.yaml)
    config = {
        'img_size': (96, 96, 96),
        'in_channels': 6,
        'out_channels': 1,
        'feature_size': 48,
        'pretrained_weights_path': None # Set to a path to test weight loading
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the model
    model = PhysicsInformedSwinUNETR(**config).to(device)
    
    # 2. Print model summary to verify the architecture
    # You can see the modified first layer here.
    print(model)
    
    # 3. Create a dummy input tensor and run a forward pass
    dummy_input = torch.randn(1, config['in_channels'], *config['img_size']).to(device)
    print(f"\nInput tensor shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output tensor shape: {output.shape}")
    
    # 4. Check if the output shape is correct
    assert output.shape == (1, config['out_channels'], *config['img_size'])
    print("\nModel instantiation and forward pass successful!")