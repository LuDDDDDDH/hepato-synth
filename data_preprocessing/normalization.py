# -*- coding: utf-8 -*-
"""
normalization.py

This script performs signal intensity normalization on the registered MRI data.

Core functionalities:
1. Uses a reference tissue (e.g., spleen or erector spinae muscle) from a 
   provided segmentation mask to establish a stable signal anchor.
2. Calculates a normalization factor from the reference tissue in a reference image
   (typically the registered portal venous phase).
3. Applies this factor to all registered images within the same session.
4. Saves the normalized images into a new 'normalization' sub-folder within
   the BIDS derivatives directory.

This step is crucial for mitigating intensity variations caused by multi-center,
multi-vendor scanner differences, ensuring that the AI model learns true
biological contrast rather than scanner-specific biases.
"""

import logging
from pathlib import Path
import ants
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import argparse

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# The modality used to calculate the reference signal intensity.
REFERENCE_IMAGE_MODALITY = "T1w_portal"
# The suffix of registered files to look for.
REGISTERED_SUFFIX = "space-portal_desc-registered"
# Label value in the segmentation mask corresponding to the reference tissue.
# This should match your segmentation protocol (e.g., 1 for liver, 2 for spleen).
REFERENCE_TISSUE_LABEL = 2 # Assuming Spleen is label 2
# Output suffix for normalized files.
OUTPUT_SUFFIX = "desc-normalized"

def find_registered_sessions(registered_root: Path) -> list:
    """Finds all valid registered session directories to process."""
    # This function is similar to the one in registration.py but looks in the derivatives.
    session_dirs = []
    for subject_dir in registered_root.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith("sub-"):
            for session_dir in subject_dir.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith("ses-"):
                    anat_dir = session_dir / "anat"
                    if anat_dir.is_dir():
                        session_dirs.append(anat_dir)
    return session_dirs

def normalize_session(registered_anat_dir: Path, mask_root: Path, output_root: Path):
    """
    Performs reference tissue normalization for a single session.
    
    Args:
        registered_anat_dir (Path): The 'anat' directory containing registered images.
        mask_root (Path): The root directory where segmentation masks are stored.
        output_root (Path): The root directory to save normalized images.
    """
    try:
        logging.info(f"Processing session for normalization: {registered_anat_dir}")
        
        # 1. Construct paths and find the corresponding mask
        subject_id = registered_anat_dir.parent.parent.name
        session_id = registered_anat_dir.parent.name
        
        # Assume mask has a similar BIDS-like naming convention, e.g., .../sub-001_ses-date_mask.nii.gz
        mask_path = mask_root / f"{subject_id}_{session_id}_segmentation.nii.gz"
        if not mask_path.exists():
            logging.warning(f"Skipping session {registered_anat_dir}: Mask not found at {mask_path}")
            return
            
        # 2. Find the reference image to calculate the normalization factor from
        ref_img_path_list = list(registered_anat_dir.glob(f"*{REFERENCE_IMAGE_MODALITY}_{REGISTERED_SUFFIX}.nii.gz"))
        if not ref_img_path_list:
            logging.warning(f"Skipping session {registered_anat_dir}: Reference image '{REFERENCE_IMAGE_MODALITY}' not found.")
            return
        ref_img_path = ref_img_path_list[0]

        # 3. Calculate the normalization factor
        logging.info("  - Calculating normalization factor from spleen...")
        ref_image_sitk = sitk.ReadImage(str(ref_img_path), sitk.sitkFloat32)
        mask_image_sitk = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        
        # Ensure mask and image are in the same physical space
        mask_image_sitk.CopyInformation(ref_image_sitk)
        
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(mask_image_sitk)
        
        if REFERENCE_TISSUE_LABEL not in stats.GetLabels():
             logging.warning(f"Skipping session {registered_anat_dir}: Reference label {REFERENCE_TISSUE_LABEL} not found in mask {mask_path}")
             return

        # Use LabelIntensityStatisticsImageFilter to get statistics from the reference tissue region
        intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
        intensity_stats_filter.Execute(mask_image_sitk, ref_image_sitk)
        
        # We use the median for robustness against outliers
        normalization_factor = intensity_stats_filter.GetMedian(REFERENCE_TISSUE_LABEL)
        
        if normalization_factor is None or normalization_factor < 1e-6:
            logging.warning(f"Skipping session {registered_anat_dir}: Invalid normalization factor ({normalization_factor}).")
            return
        
        logging.info(f"  - Normalization factor (spleen median): {normalization_factor:.2f}")

        # 4. Apply normalization to all registered images in the session
        output_dir = output_root / registered_anat_dir.relative_to(registered_anat_dir.parent.parent.parent.parent)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        registered_files = list(registered_anat_dir.glob(f"*{REGISTERED_SUFFIX}.nii.gz"))
        for img_path in tqdm(registered_files, desc="Normalizing images"):
            image_ants = ants.image_read(str(img_path))
            
            # Apply normalization: image / factor
            normalized_image_ants = image_ants / normalization_factor
            
            # Save the normalized image
            output_filename = img_path.name.replace(f"_{REGISTERED_SUFFIX}.nii.gz", f"_{OUTPUT_SUFFIX}.nii.gz")
            output_path = output_dir / output_filename
            ants.image_write(normalized_image_ants, str(output_path))
            
    except Exception as e:
        logging.error(f"Failed to normalize session {registered_anat_dir}. Error: {e}")

def main(registered_root: str, mask_root: str):
    """
    Main function to run the normalization pipeline.
    
    Args:
        registered_root (str): Path to the derivatives directory containing registered data.
        mask_root (str): Path to the directory containing segmentation masks.
    """
    registered_root = Path(registered_root)
    mask_root = Path(mask_root)
    
    # Output will be a new sub-directory in the derivatives folder
    output_root = registered_root.parent / "normalization"
    output_root.mkdir(exist_ok=True)
    
    logging.info(f"Starting normalization pipeline.")
    logging.info(f"Source Registered Data: {registered_root}")
    logging.info(f"Source Masks: {mask_root}")
    logging.info(f"Target Directory: {output_root}")
    
    session_dirs = find_registered_sessions(registered_root)
    if not session_dirs:
        logging.warning("No registered sessions found to process.")
        return
        
    for session_dir in session_dirs:
        normalize_session(session_dir, mask_root, output_root)
        
    logging.info("Normalization pipeline finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reference Tissue Intensity Normalization Tool")
    parser.add_argument("--registered_root", type=str, required=True, help="Root directory of the registered BIDS derivatives.")
    parser.add_argument("--mask_root", type=str, required=True, help="Directory containing segmentation masks.")
    
    args = parser.parse_args()
    
    main(args.registered_root, args.mask_root)