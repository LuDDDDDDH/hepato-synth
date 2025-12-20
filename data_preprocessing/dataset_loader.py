# -*- coding: utf-8 -*-
"""
dataset_loader.py

This script defines the PyTorch Dataset and DataLoader for the Hepato-Synth project,
using the powerful MONAI library for medical image data handling.

Core functionalities:
1. Creates a custom MONAI `Dataset` class to handle our preprocessed,
   multi-modal liver MRI data.
2. Constructs a file dictionary for each patient, mapping modality names to file paths.
3. Defines a chain of transformations for loading, stacking, and augmenting data.
4. Implements random patch cropping for efficient, memory-friendly training.
5. Sets up a PyTorch DataLoader to feed data batches to the GPU.
"""

import logging
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    ConcatItemsd,
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data_dicts(normalized_data_root: Path, input_keys: list, target_key: str) -> list:
    """
    Scans the preprocessed data directory and creates a list of dictionaries,
    where each dictionary represents one training sample (a patient session).
    
    Args:
        normalized_data_root (Path): Path to the 'normalization' derivatives folder.
        input_keys (list): A list of modality names to be used as input.
        target_key (str): The modality name to be used as the ground truth target.

    Returns:
        list: A list of data dictionaries, e.g.,
              [{'T1w_precontrast': 'path/to/file', 'T1w_arterial': ..., 'T1w_hbp': 'path/to/hbp'}, ...]
    """
    data_dicts = []
    
    # Assumes data is in .../normalization/sub-xxx/ses-yyy/anat/
    subject_dirs = [d for d in normalized_data_root.iterdir() if d.is_dir() and d.name.startswith('sub-')]

    for sub_dir in subject_dirs:
        for ses_dir in sub_dir.iterdir():
            if ses_dir.is_dir() and ses_dir.name.startswith('ses-'):
                anat_dir = ses_dir / "anat"
                if not anat_dir.is_dir():
                    continue
                
                file_dict = {}
                all_keys_present = True
                
                # Check if all required input and target files exist for this session
                all_modalities = input_keys + [target_key]
                for key in all_modalities:
                    file_path_list = list(anat_dir.glob(f"*{key}_desc-normalized.nii.gz"))
                    if file_path_list:
                        file_dict[key] = str(file_path_list[0])
                    else:
                        all_keys_present = False
                        logging.warning(f"Modality '{key}' not found for session {anat_dir}. Skipping this session.")
                        break
                
                if all_keys_present:
                    data_dicts.append(file_dict)
                    
    logging.info(f"Created {len(data_dicts)} data samples.")
    return data_dicts

def get_train_transforms(input_keys: list, target_key: str, patch_size: tuple) -> Compose:
    """Defines the MONAI transformation pipeline for training."""
    
    # We rename the concatenated input channels to 'image' for simplicity
    final_input_key = "image"
    
    return Compose([
        # Load all images specified by keys from the data_dict
        LoadImaged(keys=input_keys + [target_key]),
        
        # Ensure all images have a channel dimension (C, H, W, D)
        EnsureChannelFirstd(keys=input_keys + [target_key]),
        
        # Reorient all images to a standard orientation (e.g., RAS)
        Orientationd(keys=input_keys + [target_key], axcodes="RAS"),
        
        # Resample to a common voxel spacing (from config) - Optional, can be done in preprocessing
        # Spacingd(keys=input_keys + [target_key], pixdim=(1.5, 1.5, 3.0), mode=("bilinear", "nearest")),
        
        # Normalize intensity values to a [0, 1] or [-1, 1] range
        # Assuming our reference tissue normalization got us close, this is a final clip and scale
        # These values (a_min, a_max) should be determined from data exploration
        ScaleIntensityRanged(keys=input_keys + [target_key], a_min=0.0, a_max=3.0, b_min=0.0, b_max=1.0, clip=True),
        
        # Crop away empty background space to focus on the liver
        # Requires a 'liver_mask' to be present in the data_dict if used
        # CropForegroundd(keys=input_keys + [target_key], source_key=target_key),
        
        # Stack the multiple input modalities into a single multi-channel image
        ConcatItemsd(keys=input_keys, name=final_input_key, dim=0),

        # Randomly crop a patch of `patch_size` from the full image
        # This is the core of patch-based training
        # It requires a `label` key to guide cropping towards foreground if needed
        # For pure generation, we can crop randomly. Let's assume we have a liver mask for guidance.
        # Here we rename the target as the label for cropping guidance.
        # This part might need adjustment based on whether a segmentation mask is available.
        # For now, let's assume we crop randomly but ensure the patch is not empty.
        RandCropByPosNegLabeld(
            keys=[final_input_key, target_key],
            label_key=target_key, # Use target to ensure we don't just crop background
            spatial_size=patch_size,
            pos=1, # Probability of cropping a foreground patch
            neg=1, # Probability of cropping a background patch
            num_samples=4, # Extract 4 patches per image
            image_key=final_input_key,
            image_threshold=0,
        ),
        
        # Data Augmentation
        RandFlipd(keys=[final_input_key, target_key], prob=0.5, spatial_axis=0),
        RandFlipd(keys=[final_input_key, target_key], prob=0.5, spatial_axis=1),
        RandFlipd(keys=[final_input_key, target_key], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=[final_input_key, target_key], prob=0.5, max_k=3),
        
        # Ensure Tensors are of type float
        EnsureTyped(keys=[final_input_key, target_key]),
    ])

def get_dataloader(config: dict) -> DataLoader:
    """
    Creates the main DataLoader for a given study configuration.

    Args:
        config (dict): A dictionary loaded from a YAML config file.

    Returns:
        DataLoader: The PyTorch DataLoader instance.
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    # For Study 1, input and target keys are straightforward
    if config['experiment_description'].startswith("Study 1"):
        input_keys = data_cfg['input_modalities']
        target_key = data_cfg['target_modality']
        data_dicts = get_data_dicts(Path(config['preprocessed_data_root']) / "normalization", input_keys, target_key)
        
    # For Study 2 (unpaired), the dataloader is more complex (often two separate ones)
    # This is a simplified example showing how to load one domain. A real CycleGAN
    # implementation would use a more complex loader.
    elif config['experiment_description'].startswith("Study 2"):
        # This would need to be adapted for the CycleGAN structure
        # For simplicity, we'll just load domain A here.
        input_keys = data_cfg['domain_A']['input_modalities']
        target_key = input_keys[0] # No real target, just use one for transforms pipeline
        data_dicts = get_data_dicts(Path(config['preprocessed_data_root']) / "normalization", input_keys, target_key)
    
    # ... Add logic for Study 3 later
    
    else:
        raise ValueError("Unknown experiment type in config.")

    transforms = get_train_transforms(input_keys, target_key, tuple(config['data_preprocessing']['patch_size']))
    
    # Use CacheDataset to load all data into RAM for faster training, if memory allows
    # Otherwise, use the standard Dataset
    if train_cfg.get('cache_dataset', True):
        dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0, num_workers=config['num_workers'])
    else:
        dataset = Dataset(data=data_dicts, transform=transforms)
        
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader