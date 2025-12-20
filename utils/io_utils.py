# -*- coding: utf-8 -*-
"""
io_utils.py

This script provides utility functions for input/output operations, such as
loading configurations, creating experiment directories, and handling file paths.
"""

import yaml
import logging
from pathlib import Path
from datetime import datetime
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file, handling inheritance from a base config.

    If the specified config file has a 'defaults' key pointing to a base config,
    it will first load the base config and then recursively update it with the
    values from the specified config file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the loaded and merged configuration.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle inheritance
    if 'defaults' in config and config['defaults']:
        base_config_name = config['defaults'][0] # e.g., "base_config"
        base_config_path = config_path.parent / f"{base_config_name}.yaml"
        base_config = load_config(str(base_config_path))
        
        # Recursively update the base config with the specific config
        return _update_dict(base_config, config)
    else:
        return config

def _update_dict(base_dict: Dict, new_dict: Dict) -> Dict:
    """Helper function to recursively update a dictionary."""
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _update_dict(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def create_experiment_directory(config: Dict[str, Any]) -> Path:
    """
    Creates a unique directory for a new experiment based on its configuration.
    The directory structure is: output_root/experiment_name/YYYY-MM-DD_HH-MM-SS/

    Args:
        config (Dict[str, Any]): The experiment configuration dictionary.

    Returns:
        Path: The path to the created experiment directory.
    """
    output_root = Path(config['output_root'])
    
    # Use the experiment description to name the main folder, making it readable
    # Sanitize the description to be a valid folder name
    exp_name = config.get('experiment_description', 'unnamed_experiment')
    sanitized_exp_name = exp_name.replace(' ', '_').replace(':', '-').lower()
    
    # Create a timestamp for a unique sub-folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    exp_dir = output_root / sanitized_exp_name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the final configuration used for this run
    config_copy_path = exp_dir / "run_config.yaml"
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    logging.info(f"Experiment directory created: {exp_dir}")
    return exp_dir