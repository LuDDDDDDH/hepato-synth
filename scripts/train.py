# -*- coding: utf-8 -*-
"""
train.py

This is the main training script for the Hepato-Synth project.

It is a configuration-driven script that can handle the training for all three
studies defined in the research plan:
1. Study 1: Physics-Informed Accelerated Imaging (Supervised Generation)
2. Study 2: Disentangled Virtual Imaging (Unpaired GAN)
3. Study 3: Integrated Diagnostic System (Multi-task Learning)

The script orchestrates the entire training pipeline, including data loading,
model setup, the training/validation loop, metric calculation, logging, and
checkpointing.
"""

import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Any

# --- Import project modules ---
from data_preprocessing.dataset_loader import get_dataloader
from generative_models.physics_informed_unet import PhysicsInformedSwinUNETR
from generative_models.disentangled_drit import DisentangledDRIT
from generative_models.losses import AdversarialLoss, PhysicsConsistencyLoss, SSIMLoss
from diagnostic_system.multimodal_classifier import MultimodalDiagnosticSystem
from utils.io_utils import create_experiment_directory
from utils.logging_utils import setup_loggers
from utils.metrics import GenerationMetrics, DiagnosticMetrics
from utils.viz_utils import log_image_comparison
from monai.losses import DiceCELoss

def setup(config: Dict[str, Any]):
    """
    Initializes all components for the training run based on the config.
    
    Returns:
        A dictionary containing all necessary components for training.
    """
    # --- Environment Setup ---
    torch.manual_seed(config['random_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device(f"cuda:{config['gpu_ids'][0]}" if torch.cuda.is_available() and config['gpu_ids'] else "cpu")
    logging.info(f"Using device: {device}")
    
    exp_dir = create_experiment_directory(config)
    logger = setup_loggers(config, exp_dir)

    # --- Data Loading ---
    logging.info("Setting up data loaders...")
    # NOTE: The get_dataloader function needs to be extended to handle different studies.
    # We will assume it returns train_loader and val_loader based on the config.
    # For simplicity, we create them here. The real logic would be in dataset_loader.py
    # This is a placeholder for the actual data loading logic.
    # train_loader = get_dataloader(config, split='train')
    # val_loader = get_dataloader(config, split='validation')
    
    # Placeholder data loaders
    dummy_dataset = torch.utils.data.TensorDataset(torch.randn(10, 6, 96, 96, 96), torch.randn(10, 1, 96, 96, 96))
    train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=config['training']['batch_size'])
    val_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=config['training']['batch_size'])

    # --- Model Initialization ---
    logging.info(f"Initializing model: {config['model']['name']}")
    model_name = config['model']['name']
    if model_name == "PhysicsInformedSwinUNETR":
        model = PhysicsInformedSwinUNETR(**config['model'])
    elif model_name == "DisentangledDRIT":
        model = DisentangledDRIT(config['model']) # GANs have multiple optimizers, handled separately
    elif model_name == "MultimodalDiagnosticSystem":
        model = MultimodalDiagnosticSystem(config['model'])
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    model.to(device)

    # Watch model with logger (W&B specific)
    if logger:
        logger.watch_model(model)
        
    # --- Optimizers, Schedulers, Losses, and Metrics ---
    # This part is highly dependent on the study
    components = {}
    if model_name == "PhysicsInformedSwinUNETR":
        # Optimizer
        opt_cfg = config['training']['optimizer']
        optimizer = AdamW(model.parameters(), lr=opt_cfg['learning_rate'], weight_decay=opt_cfg['weight_decay'])
        components['optimizer'] = optimizer
        
        # Scheduler
        sched_cfg = config['training']['lr_scheduler']
        scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=sched_cfg['eta_min'])
        components['scheduler'] = scheduler
        
        # Losses
        components['l1_loss'] = nn.L1Loss()
        components['ssim_loss'] = SSIMLoss(spatial_dims=3)
        components['physics_loss'] = PhysicsConsistencyLoss()

        # Metrics
        components['metrics_computer'] = GenerationMetrics(device)

    # TODO: Add setup logic for Study 2 (GAN) and Study 3 (Multi-task)
    # This involves setting up multiple optimizers, more complex loss combinations etc.

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "logger": logger,
        "exp_dir": exp_dir,
        "device": device,
        "config": config,
        **components # Add all other components
    }


def train_epoch_study1(setup_dict: Dict[str, Any], epoch: int):
    """Training loop for a single epoch for Study 1."""
    model = setup_dict['model'].train()
    loader = setup_dict['train_loader']
    optimizer = setup_dict['optimizer']
    device = setup_dict['device']
    config = setup_dict['config']['training']

    # Unpack loss functions
    l1_loss_fn = setup_dict['l1_loss']
    ssim_loss_fn = setup_dict['ssim_loss']
    physics_loss_fn = setup_dict['physics_loss']
    
    # Loss weights
    w_l1 = config['loss']['components'][0]['weight']
    w_ssim = config['loss']['components'][1]['weight']
    w_phys = config['loss']['components'][2]['weight']
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for i, (input_batch, target_batch) in enumerate(pbar):
        # Assuming input_batch is (B, 6, D, H, W) and target_batch is (B, 1, D, H, W)
        inputs, targets = input_batch.to(device), target_batch.to(device)
        
        # For physics loss, we need k_hep map and liver mask
        # This data needs to be loaded by the dataloader
        # Placeholder for k_hep map and liver mask
        khep_map = inputs[:, 5:6, ...] # Assuming k_hep is the 6th channel
        liver_mask = (inputs[:, 0:1, ...] > 0).float() # A dummy mask from pre-contrast
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(inputs)
        
        # Calculate losses
        loss_l1 = l1_loss_fn(predictions, targets)
        loss_ssim = 1.0 - ssim_loss_fn(predictions, targets) # SSIM loss is 1-SSIM
        loss_phys = physics_loss_fn(predictions, khep_map, liver_mask)
        
        total_loss = (w_l1 * loss_l1) + (w_ssim * loss_ssim) + (w_phys * loss_phys)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=f"{total_loss.item():.4f}", l1=f"{loss_l1.item():.4f}", ssim=f"{loss_ssim.item():.4f}")
        
        if setup_dict['logger']:
            step = epoch * len(loader) + i
            setup_dict['logger'].log_metrics({
                "train/total_loss": total_loss.item(),
                "train/l1_loss": loss_l1.item(),
                "train/ssim_loss": loss_ssim.item(),
                "train/physics_loss": loss_phys.item(),
                "train/lr": optimizer.param_groups[0]['lr']
            }, step=step)

def validate_epoch_study1(setup_dict: Dict[str, Any], epoch: int):
    """Validation loop for a single epoch for Study 1."""
    model = setup_dict['model'].eval()
    loader = setup_dict['val_loader']
    metrics_computer = setup_dict['metrics_computer']
    device = setup_dict['device']
    
    metrics_computer.reset()
    
    with torch.no_grad():
        for input_batch, target_batch in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
            inputs, targets = input_batch.to(device), target_batch.to(device)
            
            # Here we can use SlidingWindowInferer for full volume validation
            # For simplicity, we assume patch-based validation
            predictions = model(inputs)
            
            metrics_computer.update(predictions, targets)
    
    metrics = metrics_computer.compute()
    return metrics


def run_training(config: Dict[str, Any]):
    """Main function to run the entire training pipeline."""
    
    # --- 1. Setup all components ---
    setup_dict = setup(config)
    model = setup_dict['model']
    train_loader = setup_dict['train_loader']
    val_loader = setup_dict['val_loader']
    logger = setup_dict['logger']
    exp_dir = setup_dict['exp_dir']
    
    best_metric = -1.0 # For PSNR/SSIM, higher is better

    # --- 2. Training Loop ---
    for epoch in range(config['training']['epochs']):
        # Determine which training function to call based on study
        study_name = config['experiment_description']
        if "Study 1" in study_name:
            train_epoch_study1(setup_dict, epoch)
            val_metrics = validate_epoch_study1(setup_dict, epoch)
            
            # --- Logging and Checkpointing ---
            if logger:
                logger.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
                
                # Log visual comparison
                # Get a sample batch for visualization
                sample_input, sample_target = next(iter(val_loader))
                model.eval()
                with torch.no_grad():
                    sample_pred = model(sample_input.to(setup_dict['device']))
                
                log_image_comparison(
                    epoch, exp_dir / "visuals",
                    sample_input, sample_target, sample_pred, prefix="val"
                )

            # Checkpoint saving
            current_metric = val_metrics['SSIM'] # Let's use SSIM to track the best model
            if current_metric > best_metric:
                best_metric = current_metric
                save_path = exp_dir / "checkpoints" / "best_model.pth"
                save_path.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(), save_path)
                logging.info(f"Epoch {epoch}: New best model saved to {save_path} (SSIM: {best_metric:.4f})")
        
        elif "Study 2" in study_name:
            # train_epoch_study2(...) # Requires a separate, more complex function
            pass
        elif "Study 3" in study_name:
            # train_epoch_study3(...)
            pass
            
        if 'scheduler' in setup_dict:
            setup_dict['scheduler'].step()

    if logger:
        logger.close()
    logging.info("Training finished.")

# Note: The main entry point `main.py` will call this `run_training` function.