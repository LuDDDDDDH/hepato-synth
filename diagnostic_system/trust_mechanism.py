# -*- coding: utf-8 -*-
"""
trust_mechanism.py

This script implements the final decision-making and risk-mitigation layer
for the trustworthy AI system in Study 3.

Core functionalities:
1. Takes the mean predictions and uncertainty estimations as input.
2. Implements a "circuit breaker" or "fusion" mechanism based on pre-defined
   uncertainty thresholds.
3. For segmentation, it can generate a "risk map" highlighting areas where
   the model is uncertain.
4. For classification, it can flag low-confidence predictions and suggest
   a "human-in-the-loop" review.
5. It provides a final, "trust-aware" diagnostic output.
"""

import logging
import torch
import numpy as np

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrustMechanism:
    """
    Applies decision rules to the output of the model and its uncertainty
    estimates to provide a final, trustworthy diagnostic report.
    """
    def __init__(self,
                 clf_confidence_threshold: float = 0.7,
                 clf_uncertainty_threshold: float = 0.05,
                 seg_uncertainty_threshold: float = 0.1):
        """
        Initializes the Trust Mechanism with decision thresholds.

        Args:
            clf_confidence_threshold (float): The minimum probability for the winning
                                              class to be considered a "high-confidence"
                                              prediction.
            clf_uncertainty_threshold (float): The maximum variance allowed for the
                                               winning class's probability to be
                                               considered a stable prediction.
            seg_uncertainty_threshold (float): The voxel-wise variance threshold above
                                               which a region in the generated image
                                               is considered "low-confidence".
        """
        self.clf_conf_thresh = clf_confidence_threshold
        self.clf_unc_thresh = clf_uncertainty_threshold
        self.seg_unc_thresh = seg_uncertainty_threshold
        
        logging.info("Trust Mechanism initialized with thresholds:")
        logging.info(f"  - Classification Confidence > {self.clf_conf_thresh}")
        logging.info(f"  - Classification Uncertainty < {self.clf_unc_thresh}")
        logging.info(f"  - Segmentation/Generation Uncertainty < {self.seg_unc_thresh}")

    def generate_report(self,
                        mean_clf_pred: torch.Tensor,
                        clf_uncertainty: torch.Tensor,
                        seg_uncertainty_map: torch.Tensor,
                        class_names: list) -> list:
        """
        Processes a batch of predictions and generates a list of human-readable reports.

        Args:
            mean_clf_pred (torch.Tensor): Mean classification probabilities (B, Classes).
            clf_uncertainty (torch.Tensor): Per-class variance (B, Classes).
            seg_uncertainty_map (torch.Tensor): Voxel-wise uncertainty map from the
                                                generated vHBP (B, 1, D, H, W).
            class_names (list): A list of strings for the class names.

        Returns:
            list: A list of dictionaries, where each dictionary is a structured
                  report for one sample in the batch.
        """
        batch_size = mean_clf_pred.shape[0]
        reports = []

        for i in range(batch_size):
            report = {
                "final_diagnosis": "Undetermined",
                "confidence_level": "Low",
                "warning_flags": [],
                "details": {}
            }
            
            # --- 1. Analyze Classification Confidence and Uncertainty ---
            probs = mean_clf_pred[i]
            variances = clf_uncertainty[i]
            
            winning_prob, winning_class_idx = torch.max(probs, dim=0)
            winning_class_name = class_names[winning_class_idx]
            winning_class_uncertainty = variances[winning_class_idx]
            
            report["details"]["class_probabilities"] = {name: f"{p:.4f}" for name, p in zip(class_names, probs)}
            report["details"]["class_uncertainties"] = {name: f"{v:.4f}" for name, v in zip(class_names, variances)}
            
            is_confident = winning_prob > self.clf_conf_thresh
            is_stable = winning_class_uncertainty < self.clf_unc_thresh
            
            if not is_confident:
                report["warning_flags"].append(f"LowConfidence: Max probability ({winning_prob:.2f}) is below threshold ({self.clf_conf_thresh}).")
            if not is_stable:
                 report["warning_flags"].append(f"HighUncertainty: Prediction for '{winning_class_name}' is unstable (variance={winning_class_uncertainty:.4f}).")
            
            # --- 2. Make a final decision based on the rules ---
            if is_confident and is_stable:
                report["final_diagnosis"] = winning_class_name
                report["confidence_level"] = "High"
            else:
                # If not confident, we can check the second-best guess
                # For simplicity, we just label as undetermined
                report["final_diagnosis"] = f"Undetermined (Top guess: {winning_class_name})"
            
            # --- 3. Generate the Risk Map from segmentation uncertainty ---
            # This map highlights regions where the vHBP generation might be unreliable.
            risk_map = (seg_uncertainty_map[i] > self.seg_unc_thresh).int()
            report["risk_map"] = risk_map # This tensor can be saved as a NIfTI file for visualization
            
            high_risk_voxel_ratio = torch.mean(risk_map.float())
            report["details"]["high_uncertainty_generation_ratio"] = f"{high_risk_voxel_ratio:.2%}"
            
            if high_risk_voxel_ratio > 0.1: # If more than 10% of the area is uncertain
                 report["warning_flags"].append("UnreliableGeneration: The generated virtual HBP has large areas of high uncertainty. Diagnostic features from it may be untrustworthy.")
                 
            reports.append(report)
            
        return reports

    def apply_feature_熔断(self, features: torch.Tensor, uncertainty_map: torch.Tensor) -> torch.Tensor:
        """
        A conceptual function showing how to implement the "熔断机制" (Circuit Breaker).
        This would be integrated into the diagnostic model's forward pass.
        
        It dynamically down-weights features from unreliable regions.

        Args:
            features (torch.Tensor): The feature map from a specific channel, e.g., the vHBP.
            uncertainty_map (torch.Tensor): The corresponding uncertainty map.

        Returns:
            torch.Tensor: The modulated (trust-aware) feature map.
        """
        # Create a "trust map" where high uncertainty leads to low trust (weight)
        trust_map = 1.0 - torch.clamp(uncertainty_map / self.seg_unc_thresh, 0, 1)
        
        # Apply the trust map to the features
        return features * trust_map

if __name__ == '__main__':
    # --- Example Usage & Sanity Check ---
    
    CLASS_NAMES = ["HCC", "FNH", "HCA", "ICC", "Hemangioma"]
    
    # 1. Instantiate the Trust Mechanism with default thresholds
    trust_system = TrustMechanism()
    
    # 2. Create dummy inputs (simulating the output of UncertaintyEstimator)
    # Case 1: High confidence, low uncertainty
    mean_clf_1 = torch.tensor([[0.9, 0.05, 0.02, 0.01, 0.02]])
    unc_clf_1 = torch.tensor([[0.001, 0.0005, 0.0001, 0.0001, 0.0001]])
    unc_seg_1 = torch.rand(1, 1, 64, 64, 64) * 0.05 # Low uncertainty everywhere
    
    # Case 2: Low confidence, high uncertainty
    mean_clf_2 = torch.tensor([[0.5, 0.4, 0.05, 0.02, 0.03]])
    unc_clf_2 = torch.tensor([[0.08, 0.07, 0.001, 0.001, 0.001]])
    unc_seg_2 = torch.rand(1, 1, 64, 64, 64) * 0.2 # High uncertainty everywhere
    
    # Batch them together
    batch_mean_clf = torch.cat([mean_clf_1, mean_clf_2], dim=0)
    batch_unc_clf = torch.cat([unc_clf_1, unc_clf_2], dim=0)
    batch_unc_seg = torch.cat([unc_seg_1, unc_seg_2], dim=0)
    
    # 3. Generate the reports
    reports = trust_system.generate_report(
        mean_clf_pred=batch_mean_clf,
        clf_uncertainty=batch_unc_clf,
        seg_uncertainty_map=batch_unc_seg,
        class_names=CLASS_NAMES
    )
    
    # 4. Print the reports for inspection
    import json
    print("\n--- Trust-Aware Diagnostic Reports ---")
    for i, report in enumerate(reports):
        print(f"\n--- Report for Sample {i+1} ---")
        # Don't print the large risk_map tensor
        printable_report = {k: v for k, v in report.items() if k != 'risk_map'}
        print(json.dumps(printable_report, indent=2))
        print(f"Risk map tensor shape: {report['risk_map'].shape}")

    # 5. Demonstrate the feature熔断 concept
    print("\n--- Demonstrating Feature Circuit Breaker ---")
    dummy_feature_map = torch.ones(1, 1, 4, 4, 4)
    # Create an uncertainty map where the center is highly uncertain
    dummy_unc_map = torch.zeros(1, 1, 4, 4, 4)
    dummy_unc_map[:, :, 1:3, 1:3, 1:3] = 0.2 # Above threshold of 0.1
    
    print("Original Feature Map (sum):", dummy_feature_map.sum().item())
    
    modulated_features = trust_system.apply_feature_熔断(dummy_feature_map, dummy_unc_map)
    
    print("Modulated Feature Map (center should be zeroed out):\n", modulated_features.squeeze())
    print("Modulated Feature Map (sum):", modulated_features.sum().item())
    assert modulated_features.sum() < dummy_feature_map.sum()
    print("\nFeature circuit breaker concept demonstrated successfully!")