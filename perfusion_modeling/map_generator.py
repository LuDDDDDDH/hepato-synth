# -*- coding: utf-8 -*-
"""
map_generator.py

This script serves as the master orchestrator for the entire perfusion modeling pipeline.

Core functionalities:
1. Scans a BIDS-compliant dataset of preprocessed (registered & normalized) data.
2. For each subject session, it automates the end-to-end workflow:
   a. Stacks the individual 3D MRI phase images into a single 4D volume.
   b. Calls `aif_extraction.main()` to extract the Arterial Input Function.
   c. Calls `ditc_fitter_gpu.main()` to perform GPU-accelerated model fitting,
      using the outputs from the previous steps.
3. Manages all file paths, logging, and error handling for a robust,
   batch-processing experience.

Running this single script will generate all the required physical parameter maps
for the entire dataset, preparing it for the AI model training phase.
"""

import logging
import subprocess
import sys
from pathlib import Path
import argparse
import SimpleITK as sitk
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# --- Configuration ---
# Define the order of phases to stack into a 4D image.
# This MUST match the input_modalities order expected by later scripts.
PHASE_ORDER = [
    "T1w_precontrast",
    "T1w_arterial",
    "T1w_portal",
    "T1w_equilibrium"
]

def find_sessions_to_process(normalized_root: Path) -> list:
    """Finds all valid session directories that have the required phase images."""
    session_dirs = []
    subject_dirs = [d for d in normalized_root.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    for sub_dir in subject_dirs:
        for ses_dir in sub_dir.iterdir():
            if ses_dir.is_dir() and ses_dir.name.startswith('ses-'):
                anat_dir = ses_dir / "anat"
                if not anat_dir.is_dir():
                    continue
                
                # Check if all required phases exist
                has_all_phases = all(
                    any(anat_dir.glob(f"*{phase}_desc-normalized.nii.gz")) for phase in PHASE_ORDER
                )
                if has_all_phases:
                    session_dirs.append(anat_dir)
                else:
                    logging.warning(f"Skipping session {anat_dir}: Not all required phases found.")
    return session_dirs


def stack_images_to_4d(session_anat_dir: Path, temp_dir: Path) -> Path:
    """
    Stacks the 3D phase images into a single 4D NIfTI file.

    Args:
        session_anat_dir (Path): The directory containing the 3D NIfTI files.
        temp_dir (Path): A temporary directory to store the stacked 4D file.

    Returns:
        Path: The path to the created 4D NIfTI file, or None if failed.
    """
    try:
        image_paths = []
        for phase in PHASE_ORDER:
            # Find the file for the current phase
            file_list = list(session_anat_dir.glob(f"*{phase}_desc-normalized.nii.gz"))
            if not file_list:
                raise FileNotFoundError(f"Missing phase file for {phase} in {session_anat_dir}")
            image_paths.append(str(file_list[0]))
            
        # Read images using SimpleITK
        images_sitk = [sitk.ReadImage(p, sitk.sitkFloat32) for p in image_paths]
        
        # Join the series of 3D images into a single 4D image
        joiner = sitk.JoinSeriesImageFilter()
        image_4d_sitk = joiner.Execute(images_sitk)
        
        # Create a unique name for the 4D file
        subject_id = session_anat_dir.parent.parent.name
        session_id = session_anat_dir.parent.name
        output_4d_filename = temp_dir / f"{subject_id}_{session_id}_4d_stack.nii.gz"
        
        sitk.WriteImage(image_4d_sitk, str(output_4d_filename))
        logging.info(f"  - Successfully created 4D stack: {output_4d_filename}")
        return output_4d_filename
    except Exception as e:
        logging.error(f"Failed to stack images for {session_anat_dir}: {e}")
        return None

def run_subprocess(command: list):
    """Helper function to run an external script and handle errors."""
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(process.stdout)
        if process.stderr:
            logging.warning(process.stderr)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed with exit code {e.returncode}")
        logging.error(f"Stderr: {e.stderr}")
        logging.error(f"Stdout: {e.stdout}")
        return False

def main(normalized_root: str, mask_root: str, output_root: str):
    """
    Main function to orchestrate the entire perfusion map generation pipeline.
    
    Args:
        normalized_root (str): Path to the derivatives directory with normalized data.
        mask_root (str): Path to the directory with liver masks.
        output_root (str): Root directory to save all perfusion modeling outputs.
    """
    normalized_root = Path(normalized_root)
    mask_root = Path(mask_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory for intermediate 4D files
    temp_dir = output_root / "temp_4d_stacks"
    temp_dir.mkdir(exist_ok=True)

    logging.info("Starting Batch Perfusion Map Generation Pipeline.")
    logging.info(f"Normalized Data Source: {normalized_root}")
    logging.info(f"Masks Source: {mask_root}")
    logging.info(f"Output Target: {output_root}")
    
    sessions_to_process = find_sessions_to_process(normalized_root)
    if not sessions_to_process:
        logging.warning("No valid sessions found to process.")
        return

    for session_anat_dir in tqdm(sessions_to_process, desc="Processing Sessions"):
        subject_id = session_anat_dir.parent.parent.name
        session_id = session_anat_dir.parent.name
        logging.info(f"\n===== Processing {subject_id} / {session_id} =====")
        
        # Define paths for this session
        session_output_dir = output_root / subject_id / session_id / "anat"
        session_output_dir.mkdir(parents=True, exist_ok=True)
        
        liver_mask_path = mask_root / f"{subject_id}_{session_id}_segmentation.nii.gz"
        if not liver_mask_path.exists():
            logging.error(f"Liver mask not found for {subject_id}/{session_id}. Skipping.")
            continue
        
        # --- Step 1: Stack 3D images to a 4D volume ---
        stacked_4d_path = stack_images_to_4d(session_anat_dir, temp_dir)
        if not stacked_4d_path:
            continue
            
        # --- Step 2: Run AIF Extraction ---
        logging.info("--- Running AIF Extraction ---")
        aif_command = [
            "python", "./perfusion_modeling/aif_extraction.py",
            "--input_nifti", str(stacked_4d_path),
            "--output_dir", str(session_output_dir)
        ]
        if not run_subprocess(aif_command):
            logging.error("AIF extraction failed. Skipping fitting for this session.")
            continue
        
        aif_file_path = session_output_dir / stacked_4d_path.name.replace(".nii.gz", "_aif.txt")
        if not aif_file_path.exists():
            logging.error("AIF file was not generated. Skipping fitting.")
            continue

        # --- Step 3: Run GPU-accelerated Model Fitting ---
        logging.info("--- Running DITC Model Fitting ---")
        fitter_command = [
            "python", "./perfusion_modeling/ditc_fitter_gpu.py",
            "--input_nifti", str(stacked_4d_path),
            "--aif", str(aif_file_path),
            "--mask", str(liver_mask_path),
            "--output_dir", str(session_output_dir)
        ]
        if not run_subprocess(fitter_command):
            logging.error("DITC fitting failed for this session.")
            continue
            
        logging.info(f"===== Successfully finished {subject_id} / {session_id} =====")

    # Clean up temp directory
    # for f in temp_dir.iterdir():
    #     f.unlink()
    # temp_dir.rmdir()
    logging.info("Batch processing complete. Temporary 4D stacks are kept for inspection.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master script for batch perfusion map generation.")
    parser.add_argument("--normalized_root", type=str, required=True, help="Root directory of the normalized BIDS derivatives.")
    parser.add_argument("--mask_root", type=str, required=True, help="Directory containing liver segmentation masks.")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory to save all perfusion modeling outputs.")
    
    args = parser.parse_args()
    
    main(args.normalized_root, args.mask_root, args.output_root)