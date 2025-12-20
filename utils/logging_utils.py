# -*- coding: utf-8 -*-
"""
logging_utils.py

This script provides utility functions for setting up and managing experiment
logging with tools like Weights & Biases (W&B) or TensorBoard.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

# --- Check for logger availability ---
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# --- Standard Python Logger ---
# This will be used for console and file-based logging
def setup_base_logger(log_dir: Path, log_level=logging.INFO):
    """Sets up the basic Python logger to save console output to a file."""
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    log_file = log_dir / "run.log"
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # Also print to console
        ]
    )
    logging.info(f"Console output is being saved to: {log_file}")


class ExperimentLogger:
    """A wrapper class for different experiment tracking backends."""
    def __init__(self, logger_type: str, log_dir: Path, config: Dict[str, Any]):
        self.logger_type = logger_type
        self.writer = None
        
        if self.logger_type == 'wandb':
            if not WANDB_AVAILABLE:
                logging.error("wandb is selected but not installed. Please run 'pip install wandb'.")
                self.logger_type = 'none'
            else:
                self.writer = wandb.init(
                    project=config.get('project_name', 'hepato-synth'),
                    name=f"{log_dir.parent.name}/{log_dir.name}", # e.g., study_1/timestamp
                    config=config,
                    dir=str(log_dir.parent.parent) # Store wandb files in the output_root
                )
                logging.info("Initialized Weights & Biases logger.")

        elif self.logger_type == 'tensorboard':
            if not TENSORBOARD_AVAILABLE:
                logging.error("tensorboard is selected but not available.")
                self.logger_type = 'none'
            else:
                self.writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))
                logging.info(f"Initialized TensorBoard logger. Log dir: {log_dir / 'tensorboard'}")
        else:
            logging.info("No experiment logger (wandb/tensorboard) selected.")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Logs a dictionary of metrics at a given step."""
        if self.logger_type == 'wandb':
            self.writer.log(metrics, step=step)
        elif self.logger_type == 'tensorboard':
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def log_image(self, tag: str, image_tensor, step: int):
        """Logs an image."""
        if self.logger_type == 'wandb':
            # W&B can handle PyTorch tensors directly
            self.writer.log({tag: wandb.Image(image_tensor)}, step=step)
        elif self.logger_type == 'tensorboard':
            # TensorBoard's add_image needs a tensor in (C, H, W) or (N, C, H, W) format
            self.writer.add_image(tag, image_tensor, step)
            
    def watch_model(self, model):
        """Watches a model's gradients and parameters (W&B specific)."""
        if self.logger_type == 'wandb':
            self.writer.watch(model, log='all', log_freq=100)

    def close(self):
        """Closes the logger."""
        if self.logger_type == 'wandb':
            self.writer.finish()
        elif self.logger_type == 'tensorboard':
            self.writer.close()

def setup_loggers(config: Dict[str, Any], exp_dir: Path) -> Optional[ExperimentLogger]:
    """
    Sets up both the basic file logger and the experiment tracker (W&B/TensorBoard).

    Args:
        config (Dict[str, Any]): The full experiment configuration.
        exp_dir (Path): The unique directory for this experiment run.

    Returns:
        Optional[ExperimentLogger]: An instance of the experiment logger wrapper.
    """
    setup_base_logger(exp_dir)
    
    logger_type = config.get('logging', {}).get('logger', 'none')
    
    if logger_type in ['wandb', 'tensorboard']:
        return ExperimentLogger(logger_type, exp_dir, config)
    return None