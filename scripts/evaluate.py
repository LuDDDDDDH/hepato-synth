# -*- coding: utf-8 -*-
"""
evaluate.py

This script is used for the final evaluation of a trained model on a held-out
test set.

Core functionalities:
1. Loads a trained model checkpoint and its corresponding configuration.
2. Sets up a DataLoader for the test set.
3. Performs inference on each case in the test set, typically using a
   sliding window approach for full volume prediction.
4. Computes a comprehensive set of performance metrics (e.g., PSNR, SSIM for
   generation; Dice, AUC for diagnosis).
5. Aggregates the metrics across the entire test set and saves a final report.
"""

import logging
import json
from pathlib import Path
import torch
from tqdm import tqdm
from typing import Dict, Any
import pandas as pd

# --- Import project modules ---
from data_preprocessing.dataset_loader import get_dataloader # Needs to be adapted for test set
from generative_models.physics_informed_unet import PhysicsInformedSwinUNETR
from generative_models.disentangled_drit import DisentangledDRIT
from diagnostic_system.multimodal_classifier import MultimodalDiagnosticSystem
from utils.metrics import GenerationMetrics, DiagnosticMetrics
from monai.inferers import SlidingWindowInferer

# Assuming a similar setup function as in train.py, but tailored for evaluation
def setup_for_evaluation(config: Dict[str, Any], checkpoint_path: str):
    """Initializes components needed for evaluation."""
    device = torch.device(f"cuda:{config['gpu_ids'][0]}" if torch.cuda.is_available() and config['gpu_ids'] else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Loading for Test Set ---
    # The dataloader should be configured to load the test split without shuffling or augmentation
    # test_loader = get_dataloader(config, split='test')
    
    # Placeholder test loader
    dummy_dataset = torch.utils.data.TensorDataset(torch.randn(5, 6, 128, 128, 128), torch.randn(5, 1, 128, 128, 128))
    test_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=1) # Batch size 1 for case-by-case eval

    # --- Model Initialization ---
    model_name = config['model']['name']
    if model_name == "PhysicsInformedSwinUNETR":
        model = PhysicsInformedSwinUNETR(**config['model'])
    # ... Add other model types
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    model.to(device)

    # --- Load Trained Weights ---
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # --- Metrics Computer ---
    if "Study 1" in config['experiment_description']:
        metrics_computer = GenerationMetrics(device)
    # ... Add other metric types
    else:
         raise ValueError("Metrics for this study type not implemented yet.")
         
    # --- Sliding Window Inferer ---
    # This is crucial for evaluating on full-sized images
    inferer = SlidingWindowInferer(
        roi_size=config['data_preprocessing']['patch_size'],
        sw_batch_size=4, # Process 4 patches at a time
        overlap=0.5,     # Overlap between patches to reduce boundary artifacts
        mode="gaussian"  # Use Gaussian weighting for smoother blending
    )
    
    return {
        "model": model,
        "test_loader": test_loader,
        "metrics_computer": metrics_computer,
        "inferer": inferer,
        "device": device
    }

def run_evaluation(config: Dict[str, Any], checkpoint_path: str, output_dir: Path):
    """Main function to run the evaluation pipeline."""
    
    setup_dict = setup_for_evaluation(config, checkpoint_path)
    model = setup_dict['model']
    test_loader = setup_dict['test_loader']
    metrics_computer = setup_dict['metrics_computer']
    inferer = setup_dict['inferer']
    device = setup_dict['device']
    
    metrics_computer.reset()
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs, targets = batch_data[0].to(device), batch_data[1].to(device)
            
            # Use the inferer for full-volume prediction
            predictions = inferer(inputs, model)
            
            # Update metrics computer
            # Note: The specific call might vary based on the metric computer's needs
            metrics_computer.update(predictions, targets)

    # --- Compute and Report Final Metrics ---
    final_metrics = metrics_computer.compute()
    logging.info("\n--- Final Evaluation Results ---")
    
    # Convert to DataFrame for nice printing and easy saving
    metrics_df = pd.DataFrame([final_metrics])
    print(metrics_df.to_string())
    
    # Save the report
    report_path = output_dir / "evaluation_report.csv"
    metrics_df.to_csv(report_path, index=False)
    logging.info(f"\nEvaluation report saved to: {report_path}")

# Note: The main entry point `main.py` will call this function.