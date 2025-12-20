# -*- coding: utf-8 -*-
"""
losses.py

This script defines all loss functions used in the Hepato-Synth project,
including standard pixel-wise losses, GAN losses, and custom, physics-informed losses.

Decoupling loss functions into this module allows for flexible combination and
experimentation via the configuration files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import SSIMLoss

# ==============================================================================
# Base Loss Wrapper for easy combination
# ==============================================================================

class LossFromConfig(nn.Module):
    """
    A wrapper class to dynamically create a composite loss function
    from a configuration dictionary.
    """
    def __init__(self, loss_config: list):
        super().__init__()
        self.components = nn.ModuleList()
        self.weights = []
        
        for comp_cfg in loss_config:
            name = comp_cfg['name']
            weight = comp_cfg['weight']
            params = comp_cfg.get('params', {})
            
            if name == "L1Loss":
                loss_fn = nn.L1Loss(**params)
            elif name == "SSIMLoss":
                loss_fn = SSIMLoss(spatial_dims=3, **params)
            elif name == "PhysicsConsistencyLoss":
                loss_fn = PhysicsConsistencyLoss(**params)
            elif name == "AdversarialLoss":
                loss_fn = AdversarialLoss(**params)
            # Add other custom losses here
            else:
                raise ValueError(f"Unknown loss component: {name}")
                
            self.components.append(loss_fn)
            self.weights.append(weight)

    def forward(self, *args, **kwargs):
        total_loss = 0.0
        for weight, loss_fn in zip(self.weights, self.components):
            # Each loss function might take different arguments
            # We will pass all arguments and let the specific loss function
            # pick what it needs. This is a flexible but requires careful design.
            # A more robust way is to define what each loss needs.
            # For now, we assume a simple signature.
            # A better approach for multiple complex losses:
            # The training loop calls each loss function with its required inputs
            # and then combines them. This wrapper is for simple cases.
            
            # This wrapper is more for holding the modules. The training loop
            # should handle the logic of passing the correct tensors.
            # Example logic in trainer:
            # l1_loss = self.loss.components[0](pred, target)
            # phys_loss = self.loss.components[1](pred, khep_map)
            # total_loss = w1 * l1_loss + w2 * phys_loss
            pass # The main calculation will happen in the training script.

# ==============================================================================
# Study 1: Physics Consistency Loss
# ==============================================================================

class PhysicsConsistencyLoss(nn.Module):
    """
    Ensures the generated HBP signal intensity is consistent with the
    underlying liver function as represented by the k_hep parameter map.

    The core idea is that, within the liver, higher k_hep (hepatocyte uptake rate)
    should correlate with higher signal intensity in the generated HBP.
    We enforce this by maximizing the Pearson correlation coefficient between
    the two maps within the liver mask.
    """
    def __init__(self, liver_mask_key='liver_mask'):
        super().__init__()
        self.liver_mask_key = liver_mask_key

    def forward(self, generated_hbp: torch.Tensor, khep_map: torch.Tensor, liver_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated_hbp (torch.Tensor): The AI-generated HBP image batch (B, 1, D, H, W).
            khep_map (torch.Tensor): The corresponding k_hep physical map (B, 1, D, H, W).
            liver_mask (torch.Tensor): The binary liver mask (B, 1, D, H, W).
        
        Returns:
            torch.Tensor: The loss value. Loss is 1 - correlation, so we minimize it.
        """
        batch_size = generated_hbp.shape[0]
        total_corr_loss = 0.0

        for i in range(batch_size):
            hbp_slice = generated_hbp[i].squeeze()
            khep_slice = khep_map[i].squeeze()
            mask_slice = liver_mask[i].squeeze().bool()

            # Select voxels within the liver mask
            hbp_masked = hbp_slice[mask_slice]
            khep_masked = khep_slice[mask_slice]
            
            if hbp_masked.numel() < 10: # If mask is too small, skip
                continue

            # Center the variables (subtract the mean)
            hbp_centered = hbp_masked - hbp_masked.mean()
            khep_centered = khep_masked - khep_masked.mean()

            # Calculate covariance and standard deviations
            covariance = (hbp_centered * khep_centered).sum()
            hbp_std = torch.sqrt((hbp_centered**2).sum())
            khep_std = torch.sqrt((khep_centered**2).sum())

            # Calculate Pearson correlation coefficient
            # Add epsilon to avoid division by zero
            correlation = covariance / (hbp_std * khep_std + 1e-6)
            
            # We want to maximize correlation, which is equivalent to minimizing (1 - correlation)
            total_corr_loss += (1.0 - correlation)
            
        return total_corr_loss / batch_size

# ==============================================================================
# Study 2: GAN-related Losses
# ==============================================================================

class AdversarialLoss(nn.Module):
    """
    Adversarial Loss for the discriminator, based on Least Squares GAN (LSGAN).
    LSGAN is more stable than traditional GAN with sigmoid cross-entropy.

    - For discriminator: Minimize (D(real) - 1)^2 + (D(fake) - 0)^2
    - For generator:   Minimize (D(fake) - 1)^2
    """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Args:
            prediction (torch.Tensor): The output of the discriminator.
            is_real (bool): Whether the input to the discriminator was a real or fake image.
        
        Returns:
            torch.Tensor: The LSGAN loss value.
        """
        if is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        
        # Expand target tensor to match the size of the prediction (e.g., for PatchGAN)
        return self.loss(prediction, target_tensor.expand_as(prediction))

# Note: CycleConsistencyLoss and IdentityLoss are typically just L1Loss,
# so we can reuse nn.L1Loss directly in the training script for simplicity and clarity.
# For example:
# loss_cycle_A = l1_loss(reconstructed_A, real_A)
# loss_identity_A = l1_loss(identity_A, real_A)