# -*- coding: utf-8 -*-
"""
viz_utils.py

This script provides utility functions for visualizing 3D medical imaging data,
primarily for logging and debugging purposes during training and inference.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_slice_as_png(tensor_3d: torch.Tensor, 
                      output_path: Path, 
                      title: str = "", 
                      slice_dim: int = 0, 
                      slice_idx: int = -1):
    """
    Saves a central slice of a 3D tensor as a PNG image.

    Args:
        tensor_3d (torch.Tensor): A 3D tensor (D, H, W). Assumes no channel dim.
        output_path (Path): The full path to save the PNG file.
        title (str): The title to display on the plot.
        slice_dim (int): The dimension along which to slice (0 for depth, 1 for height, 2 for width).
        slice_idx (int): The index of the slice to save. If -1, the central slice is used.
    """
    if not isinstance(tensor_3d, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if tensor_3d.ndim != 3:
        raise ValueError("Input tensor must be 3D.")
        
    # Move tensor to CPU and convert to numpy
    tensor_3d = tensor_3d.cpu().detach().numpy()
    
    # Select the slice
    if slice_idx == -1:
        slice_idx = tensor_3d.shape[slice_dim] // 2
    
    if slice_dim == 0:
        slice_data = tensor_3d[slice_idx, :, :]
    elif slice_dim == 1:
        slice_data = tensor_3d[:, slice_idx, :]
    elif slice_dim == 2:
        slice_data = tensor_3d[:, :, slice_idx]
    else:
        raise ValueError("slice_dim must be 0, 1, or 2.")
        
    # Plot and save
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(slice_data, cmap='gray', origin='lower')
    ax.set_title(title, fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close(fig)


def log_image_comparison(
    epoch: int,
    output_dir: Path,
    input_tensor: torch.Tensor, # The multi-channel input
    target_tensor: torch.Tensor, # The ground truth
    prediction_tensor: torch.Tensor, # The model's output
    prefix: str = "val"
):
    """
    Logs a visual comparison of input, target, and prediction tensors for one sample.
    Saves a composite image of the central axial slices.

    Args:
        epoch (int): The current epoch number.
        output_dir (Path): The directory to save the output image.
        input_tensor (torch.Tensor): A single sample from the input batch (C, D, H, W).
        target_tensor (torch.Tensor): The ground truth tensor (1, D, H, W).
        prediction_tensor (torch.Tensor): The model's prediction (1, D, H, W).
        prefix (str): A prefix for the filename (e.g., 'train' or 'val').
    """
    # Select the first sample from the batch (if it is a batch)
    if input_tensor.ndim == 5: input_tensor = input_tensor[0]
    if target_tensor.ndim == 5: target_tensor = target_tensor[0]
    if prediction_tensor.ndim == 5: prediction_tensor = prediction_tensor[0]

    # Let's visualize the first MRI channel (pre-contrast) and the target vs prediction
    t1_pre = input_tensor[0] # C=0
    target = target_tensor[0]
    prediction = prediction_tensor[0]
    
    # Get the central slice index from the depth dimension
    central_slice_idx = t1_pre.shape[0] // 2
    
    # Prepare data for plotting
    t1_slice = t1_pre[central_slice_idx].cpu().detach().numpy()
    target_slice = target[central_slice_idx].cpu().detach().numpy()
    pred_slice = prediction[central_slice_idx].cpu().detach().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im1 = axes[0].imshow(t1_slice, cmap='gray', origin='lower')
    axes[0].set_title("Input (T1 Pre-contrast)", fontsize=14)
    axes[0].axis('off')
    
    im2 = axes[1].imshow(target_slice, cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth HBP", fontsize=14)
    axes[1].axis('off')
    
    im3 = axes[2].imshow(pred_slice, cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[2].set_title("Model Prediction HBP", fontsize=14)
    axes[2].axis('off')
    
    fig.suptitle(f"Epoch {epoch} - Visual Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_filename = output_dir / f"{prefix}_epoch_{epoch:04d}_comparison.png"
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_filename, dpi=200)
    plt.close(fig)