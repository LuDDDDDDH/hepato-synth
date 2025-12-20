# -*- coding: utf-8 -*-
"""
metrics.py

This script defines metric calculation classes for all tasks in the project,
ensuring consistent and standardized evaluation.

It leverages the efficient implementations from `torchmetrics` for generation
tasks and `monai.metrics` for segmentation and classification tasks.
"""
from typing import List
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from monai.metrics import DiceMetric, ROCAUCMetric

class GenerationMetrics:
    """A class to compute and manage metrics for image generation tasks."""
    def __init__(self, device: torch.device):
        self.device = device
        # PSNR metric. data_range is the possible range of input values.
        # Assuming images are normalized to [0, 1].
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        # SSIM metric.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update the state with new predictions and targets."""
        # torchmetrics expects (N, C, H, W, D) format
        self.psnr.update(preds, target)
        self.ssim.update(preds, target)
        
    def compute(self) -> dict:
        """Compute the final metrics."""
        return {
            "PSNR": self.psnr.compute().item(),
            "SSIM": self.ssim.compute().item(),
        }
        
    def reset(self):
        """Reset the state of the metrics."""
        self.psnr.reset()
        self.ssim.reset()

class DiagnosticMetrics:
    """A class to compute and manage metrics for diagnostic tasks (seg & class)."""
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        num_classes = len(class_names)
        
        # Dice metric for segmentation. `include_background=False` is common.
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        
        # AUC metric for classification.
        self.auc_metric = ROCAUCMetric()

    def update(self, seg_preds: torch.Tensor, seg_target: torch.Tensor, 
               clf_preds: torch.Tensor, clf_target: torch.Tensor):
        """
        Update the state with new predictions and targets for both tasks.
        
        Args:
            seg_preds (torch.Tensor): Segmentation predictions (logits or probs).
            seg_target (torch.Tensor): Segmentation ground truth (one-hot format).
            clf_preds (torch.Tensor): Classification predictions (probs).
            clf_target (torch.Tensor): Classification ground truth (integer labels).
        """
        # For Dice, we need to convert logits to one-hot predictions
        seg_preds_one_hot = torch.argmax(seg_preds, dim=1, keepdim=True)
        seg_preds_one_hot = torch.nn.functional.one_hot(seg_preds_one_hot.squeeze(1), num_classes=seg_target.shape[1])
        seg_preds_one_hot = seg_preds_one_hot.permute(0, 4, 1, 2, 3) # Move channel to dim 1
        
        self.dice_metric.update(y_pred=seg_preds_one_hot, y=seg_target)
        
        # For AUC, target needs to be one-hot
        clf_target_one_hot = torch.nn.functional.one_hot(clf_target, num_classes=clf_preds.shape[1])
        self.auc_metric.update(y_pred=clf_preds, y=clf_target_one_hot)

    def compute(self) -> dict:
        """Compute the final metrics."""
        # Compute mean Dice across all classes and batches
        mean_dice = self.dice_metric.aggregate().item()
        
        # Compute mean AUC across all classes and batches
        mean_auc = self.auc_metric.aggregate()
        if isinstance(mean_auc, torch.Tensor):
             mean_auc = mean_auc.item()

        return {
            "Mean_Dice": mean_dice,
            "Mean_AUC": mean_auc,
        }
        
    def reset(self):
        """Reset the state of the metrics."""
        self.dice_metric.reset()
        self.auc_metric.reset()