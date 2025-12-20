# -*- coding: utf-8 -*-
"""
registration.py

This script performs 4D non-rigid registration on the time-series MRI data 
organized in a BIDS-like structure.

Core functionalities:
1. Identifies a fixed (reference) image, typically the portal venous phase.
2. Identifies moving images (all other phases like pre-contrast, arterial, etc.).
3. Uses the powerful ANTsPy library to perform non-rigid registration 
   (SyN algorithm) for each moving image to the fixed image.
4. Saves the registered images and transformation fields into a BIDS-compliant 
   'derivatives' directory.

This step is crucial for eliminating respiratory motion artifacts and ensuring
voxel-wise correspondence across all temporal phases, as required for accurate
pharmacokinetic modeling and AI training (as per Study Plan Phase 1).
"""

import os
import logging
from pathlib import Path
import ants
from tqdm import tqdm
import argparse

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# The modality name that will be used as the fixed reference for registration.
FIXED_IMAGE_MODALITY = "T1w_portal"
# Suffix to add to the output files for clarity.
OUTPUT_SUFFIX = "space-portal_desc-registered"

def find_session_dirs(bids_root: Path) -> list:
    """Finds all valid session directories (e.g., .../sub-xxx/ses-yyy/anat) to process."""
    session_dirs = []
    for subject_dir in bids_root.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith("sub-"):
            for session_dir in subject_dir.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith("ses-"):
                    anat_dir = session_dir / "anat"
                    if anat_dir.is_dir():
                        session_dirs.append(anat_dir)
    return session_dirs


def register_session(session_anat_dir: Path, derivatives_dir: Path):
    """
    Performs registration for all images within a single session directory.
    
    Args:
        session_anat_dir (Path): The 'anat' directory of a specific session.
        derivatives_dir (Path): The root directory for BIDS derivatives.
    """
    try:
        logging.info(f"Processing session: {session_anat_dir}")

        # 1. Find the fixed image
        fixed_image_path_list = list(session_anat_dir.glob(f"*{FIXED_IMAGE_MODALITY}.nii.gz"))
        if not fixed_image_path_list:
            logging.warning(f"Skipping session {session_anat_dir}: Fixed image '{FIXED_IMAGE_MODALITY}' not found.")
            return
        fixed_image_path = fixed_image_path_list[0]
        logging.info(f"  - Found Fixed Image: {fixed_image_path.name}")
        fixed_image = ants.image_read(str(fixed_image_path))

        # 2. Find all moving images
        all_nii_files = list(session_anat_dir.glob("*.nii.gz"))
        moving_image_paths = [p for p in all_nii_files if p != fixed_image_path]
        if not moving_image_paths:
            logging.warning(f"  - No moving images found in {session_anat_dir}. Nothing to do.")
            return

        # 3. Create output directory structure in derivatives
        # e.g., .../derivatives/registration/sub-001/ses-20231101/anat/
        relative_path = session_anat_dir.relative_to(session_anat_dir.parent.parent.parent)
        output_dir = derivatives_dir / "registration" / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Also copy the fixed image to the output directory for completeness
        fixed_output_name = fixed_image_path.name.replace(".nii.gz", f"_{OUTPUT_SUFFIX}.nii.gz")
        ants.image_write(fixed_image, str(output_dir / fixed_output_name))

        # 4. Loop through moving images and register them
        for moving_path in tqdm(moving_image_paths, desc="Registering phases"):
            logging.info(f"  - Registering: {moving_path.name}")
            moving_image = ants.image_read(str(moving_path))

            # Perform non-rigid registration using SyN
            # 'SyN' is a high-performance algorithm suitable for this task.
            # 't': Rigid, 'a': Affine, 's': SyN (non-rigid)
            # grad_step=0.1 is a standard parameter.
            # We use Mutual Information ('MI') as the metric, which is robust
            # for multi-modal or contrast-changed images.
            transform = ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                type_of_transform='SyN',
                grad_step=0.1,
                metric='MI',
                reg_iterations=(100, 70, 50, 20),
                syn_metric='MI',
                syn_sampling=32,
                verbose=False
            )

            # Apply the transformation to the moving image
            warped_image = ants.apply_transforms(
                fixed=fixed_image,
                moving=moving_image,
                transformlist=transform['fwdtransforms'],
                interpolator='linear' # Use linear interpolation
            )
            
            # Save the warped image
            output_filename = moving_path.name.replace(".nii.gz", f"_{OUTPUT_SUFFIX}.nii.gz")
            output_path = output_dir / output_filename
            ants.image_write(warped_image, str(output_path))
            logging.info(f"    -> Saved registered image to {output_path}")

            # Optionally, save the transformation files as well for quality control
            # transform_prefix = output_filename.replace(".nii.gz", "_transform")
            # os.rename(transform['fwdtransforms'][0], output_dir / f"{transform_prefix}_1Warp.nii.gz")
            # os.rename(transform['fwdtransforms'][1], output_dir / f"{transform_prefix}_0GenericAffine.mat")

    except Exception as e:
        logging.error(f"Failed to process session {session_anat_dir}. Error: {e}")


def main(bids_root: str):
    """
    Main function to run the registration pipeline for all sessions in a BIDS directory.
    
    Args:
        bids_root (str): Path to the BIDS-structured data directory.
    """
    bids_root = Path(bids_root)
    if not bids_root.is_dir():
        logging.error(f"BIDS root directory not found: {bids_root}")
        return

    # All outputs will be saved in a 'derivatives' folder at the top level
    derivatives_dir = bids_root.parent / f"{bids_root.name}_derivatives"
    derivatives_dir.mkdir(exist_ok=True)
    
    logging.info(f"Starting registration pipeline.")
    logging.info(f"BIDS Source: {bids_root}")
    logging.info(f"Derivatives Target: {derivatives_dir}")

    # Find all sessions to be processed
    session_dirs_to_process = find_session_dirs(bids_root)
    if not session_dirs_to_process:
        logging.warning("No valid session/anat directories found to process.")
        return

    logging.info(f"Found {len(session_dirs_to_process)} sessions to process.")
    
    # Process each session
    for session_dir in session_dirs_to_process:
        register_session(session_dir, derivatives_dir)
        
    logging.info("Registration pipeline finished.")


if __name__ == '__main__':
    # --- Example Usage ---
    # python data_preprocessing/registration.py --bids_root /path/to/output/bids
    
    parser = argparse.ArgumentParser(description="4D Non-rigid Registration Tool for BIDS data")
    parser.add_argument("--bids_root", type=str, required=True, help="Root directory of the BIDS-structured NIfTI files.")
    
    args = parser.parse_args()
    
    main(args.bids_root)