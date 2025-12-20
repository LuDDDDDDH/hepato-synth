# -*- coding: utf-8 -*-
"""
uncertainty_quantification.py

This script implements the mechanism for quantifying model uncertainty using
Monte Carlo Dropout (MC Dropout), as proposed for the trustworthy AI system in Study 3.

Core functionalities:
1. Defines an `UncertaintyEstimator` class that takes a trained model.
2. During inference, it activates the model's dropout layers (`model.train()`).
3. It performs multiple stochastic forward passes (N times) on the same input data.
4. It calculates the predictive uncertainty by computing the variance across these
   N predictions for both the segmentation and classification tasks.
5. Outputs include the mean prediction and a corresponding uncertainty map/value.
"""

import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .multimodal_classifier import MultimodalDiagnosticSystem

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UncertaintyEstimator:
    """
    A wrapper class to estimate predictive uncertainty for a given multi-task
    diagnostic model using Monte Carlo Dropout.
    """
    def __init__(self, model: MultimodalDiagnosticSystem, num_samples: int = 25):
        """
        Initializes the Uncertainty Estimator.

        Args:
            model (MultimodalDiagnosticSystem): The trained diagnostic model which must
                                                contain dropout layers.
            num_samples (int): The number of stochastic forward passes to perform (N).
                               A higher number gives a better estimate but is slower.
        """
        if num_samples < 2:
            raise ValueError("num_samples for MC Dropout must be at least 2.")
            
        self.model = model
        self.num_samples = num_samples
        logging.info(f"Uncertainty Estimator initialized with N={self.num_samples} MC samples.")

    @torch.no_grad()
    def estimate(self, input_tensor: torch.Tensor) -> dict:
        """
        Performs MC Dropout to estimate uncertainty for a batch of inputs.

        Args:
            input_tensor (torch.Tensor): The input batch tensor (B, C, D, H, W).

        Returns:
            dict: A dictionary containing the mean predictions and their uncertainties.
                'seg_mean': Mean segmentation prediction (B, Classes, D, H, W)
                'seg_uncertainty': Voxel-wise predictive variance (B, 1, D, H, W)
                'clf_mean': Mean classification probabilities (B, Classes)
                'clf_uncertainty': Per-class variance (B, Classes)
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # --- Crucial step: Activate Dropout layers for inference ---
        self.model.train()

        # --- Store predictions from each forward pass ---
        all_seg_preds = []
        all_clf_preds = []

        logging.info(f"Performing {self.num_samples} stochastic forward passes...")
        for _ in tqdm(range(self.num_samples), desc="MC Dropout Sampling"):
            seg_logits, clf_logits = self.model(input_tensor)
            
            # We compute uncertainty on the probabilities (post-softmax)
            seg_probs = F.softmax(seg_logits, dim=1)
            clf_probs = F.softmax(clf_logits, dim=1)
            
            all_seg_preds.append(seg_probs.cpu())
            all_clf_preds.append(clf_probs.cpu())
            
        # Stack the predictions along a new dimension for easy computation
        # Shape: (N, B, Classes, D, H, W) for segmentation
        # Shape: (N, B, Classes) for classification
        all_seg_preds = torch.stack(all_seg_preds)
        all_clf_preds = torch.stack(all_clf_preds)
        
        # --- Calculate Mean and Variance ---
        
        # 1. Segmentation
        # Mean prediction is the average of the probabilities
        mean_seg_pred = torch.mean(all_seg_preds, dim=0)
        
        # Predictive variance is the variance of the probabilities.
        # A common way to get a single uncertainty map is to compute the entropy
        # of the mean prediction, or the sum of variances across classes.
        # Let's use the variance of the winning class probability as a simple,
        # interpretable metric. Or total variance.
        # Total variance is a good summary measure.
        seg_variance = torch.var(all_seg_preds, dim=0)
        # Sum variance across all classes to get a single voxel-wise uncertainty value
        seg_total_variance = torch.sum(seg_variance, dim=1, keepdim=True)
        
        # 2. Classification
        mean_clf_pred = torch.mean(all_clf_preds, dim=0)
        clf_variance = torch.var(all_clf_preds, dim=0)

        # --- Set model back to eval mode ---
        self.model.eval()
        
        return {
            "seg_mean": mean_seg_pred.to(device),
            "seg_uncertainty": seg_total_variance.to(device),
            "clf_mean": mean_clf_pred.to(device),
            "clf_uncertainty": clf_variance.to(device)
        }

if __name__ == '__main__':
    # --- Example Usage & Sanity Check ---
    
    # These parameters would come from study_3_diagnostic.yaml
    config = {
        'backbone': {
            'img_size': (64, 64, 64), # Use smaller size for quick test
            'in_channels': 7,
            'out_channels_seg': 6,
            'feature_size': 48,
            'pretrained_weights_path': None
        },
        'classification_head': {
            'hidden_features': 512,
            'out_features': 5,
            'dropout_rate': 0.5 # Dropout MUST be > 0 for MC Dropout to work
        }
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate a model with dropout
    model = MultimodalDiagnosticSystem(config).to(device)
    
    # 2. Instantiate the Uncertainty Estimator
    # Use a small number of samples for a quick test
    estimator = UncertaintyEstimator(model, num_samples=10)
    
    # 3. Create a dummy input tensor
    dummy_input = torch.randn(1, config['backbone']['in_channels'], *config['backbone']['img_size']).to(device)
    print(f"Input tensor shape: {dummy_input.shape}")
    
    # 4. Get the estimations
    results = estimator.estimate(dummy_input)
    
    # 5. Check the output shapes and content
    print("\n--- Uncertainty Estimation Results ---")
    for key, tensor in results.items():
        print(f"{key}:")
        print(f"  - Shape: {tensor.shape}")
        print(f"  - Device: {tensor.device}")
        print(f"  - Min value: {tensor.min().item():.4f}")
        print(f"  - Max value: {tensor.max().item():.4f}")
        print(f"  - Mean value: {tensor.mean().item():.4f}")
        
    # Check shapes
    assert results['seg_mean'].shape == (1, config['backbone']['out_channels_seg'], *config['backbone']['img_size'])
    assert results['seg_uncertainty'].shape == (1, 1, *config['backbone']['img_size'])
    assert results['clf_mean'].shape == (1, config['classification_head']['out_features'])
    assert results['clf_uncertainty'].shape == (1, config['classification_head']['out_features'])

    print("\nUncertainty estimator instantiation and forward pass successful!")