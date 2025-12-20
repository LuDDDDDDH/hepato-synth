# -*- coding: utf-8 -*-
"""
main.py

The single command-line entry point for the Hepato-Synth project.

This script acts as a dispatcher, parsing user commands and configurations,
and then launching the appropriate workflow (training, evaluation, or inference)
by calling the corresponding modules from the `scripts/` directory.

Example Usages:
----------------
1. To train a model for Study 1:
   python main.py --mode train --config configs/study_1_acceleration.yaml

2. To evaluate the best model from a Study 1 training run:
   python main.py --mode evaluate \
                  --config /path/to/outputs/study_1.../run_config.yaml \
                  --checkpoint /path/to/outputs/study_1.../checkpoints/best_model.pth

3. To run inference on a new case with a trained Study 1 model:
   python main.py --mode inference \
                  --config /path/to/outputs/study_1.../run_config.yaml \
                  --checkpoint /path/to/outputs/study_1.../checkpoints/best_model.pth \
                  --input_dir /path/to/new_case_data \
                  --output_dir /path/to/inference_results
"""

import argparse
import logging
from pathlib import Path

# --- Import project modules ---
# We use a try-except block to provide helpful import error messages.
try:
    from utils.io_utils import load_config
    # We import the scripts dynamically inside the main function to keep startup fast
    # and dependencies clean for different modes.
except ImportError as e:
    print("Error: Failed to import necessary project modules.")
    print("Please ensure you are running this script from the project's root directory")
    print("and all required packages are installed.")
    print(f"Import Error: {e}")
    exit(1)

# --- Basic Logging Setup for the main entry point ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to parse arguments and dispatch to the correct workflow.
    """
    parser = argparse.ArgumentParser(
        description="Hepato-Synth: A Physics-Informed AI Pipeline for Liver MRI.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    
    # --- Core Arguments ---
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['train', 'evaluate', 'inference'],
        help="The operational mode to run the script in.\n"
             "  - train: Start a new training session from a config file.\n"
             "  - evaluate: Evaluate a trained model on a test set.\n"
             "  - inference: Run a trained model on a single new case."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file. For evaluation or inference,\n"
             "it's best to use the `run_config.yaml` from the experiment's output directory."
    )
    
    # --- Arguments for Evaluation & Inference ---
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the model checkpoint (.pth file) to load.\n"
             "(Required for 'evaluate' and 'inference' modes)."
    )
    
    # --- Arguments for Inference ---
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input directory for a single case.\n"
             "(Required for 'inference' mode)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the directory where inference results will be saved.\n"
             "(Required for 'inference' mode)."
    )

    args = parser.parse_args()
    
    # --- Argument Validation ---
    if args.mode in ['evaluate', 'inference'] and not args.checkpoint:
        parser.error("--checkpoint is required for 'evaluate' and 'inference' modes.")
    
    if args.mode == 'inference' and (not args.input_dir or not args.output_dir):
        parser.error("--input_dir and --output_dir are required for 'inference' mode.")

    # --- Load Configuration ---
    logging.info(f"Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {args.config}. Aborting.")
        return
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}. Aborting.")
        return

    # --- Mode Dispatching ---
    logging.info(f"Starting in '{args.mode.upper()}' mode...")
    
    if args.mode == 'train':
        try:
            from scripts.train import run_training
            run_training(config)
        except ImportError:
            logging.error("Could not import 'run_training' from 'scripts.train'.")
            
    elif args.mode == 'evaluate':
        try:
            from scripts.evaluate import run_evaluation
            # The output of the evaluation will be saved relative to the checkpoint's folder
            checkpoint_path = Path(args.checkpoint)
            eval_output_dir = checkpoint_path.parent.parent / "evaluation"
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            run_evaluation(config, str(checkpoint_path), eval_output_dir)
        except ImportError:
            logging.error("Could not import 'run_evaluation' from 'scripts.evaluate'.")

    elif args.mode == 'inference':
        try:
            from scripts.inference import run_inference
            run_inference(config, args.checkpoint, Path(args.input_dir), Path(args.output_dir))
        except ImportError:
            logging.error("Could not import 'run_inference' from 'scripts.inference'.")
    
    else:
        # This case should not be reached due to `choices` in argparse
        logging.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()