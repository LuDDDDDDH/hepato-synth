# -*- coding: utf-8 -*-
"""
dicom_parser.py

This script automates the conversion and organization of raw DICOM data from a 
clinical source into a standardized, BIDS-like directory structure.

Core functionalities:
1. Traverses a source directory of DICOM files.
2. Groups DICOMs by patient, study, and series using metadata.
3. Converts each DICOM series into a NIfTI (.nii.gz) file.
4. Intelligently renames NIfTI files to a standard modality name 
   (e.g., T1w_precontrast) based on DICOM SeriesDescription and predefined rules.
5. Organizes the output into a `sub-<PatientID>/ses-<StudyDate>/anat/` structure.

This is the foundational first step in building the standardized clinical cohort
as described in Phase 1 of the research plan.
"""

import os
import re
import logging
from pathlib import Path
from collections import defaultdict
import pydicom
import dicom2nifti
from tqdm import tqdm

# --- Configuration for Series Description Mapping ---
# This is the "brain" of the parser. It uses regular expressions to map the
# highly variable SeriesDescription from DICOM headers to our clean, standard names.
# This dictionary needs to be customized based on the actual DICOMs from your hospital.
SERIES_MAP = {
    "T1w_precontrast": [
        re.compile(r'.*pre.*', re.IGNORECASE),
        re.compile(r'.*unenhanced.*', re.IGNORECASE),
        re.compile(r'.*plain.*', re.IGNORECASE),
    ],
    "T1w_arterial": [
        re.compile(r'.*art.*', re.IGNORECASE),
        re.compile(r'.*arterial.*', re.IGNORECASE),
    ],
    "T1w_portal": [
        re.compile(r'.*pv.*', re.IGNORECASE),
        re.compile(r'.*portal.*', re.IGNORECASE),
        re.compile(r'.*venous.*', re.IGNORECASE),
    ],
    "T1w_equilibrium": [
        re.compile(r'.*del.*', re.IGNORECASE),
        re.compile(r'.*delay.*', re.IGNORECASE),
        re.compile(r'.*equilib.*', re.IGNORECASE),
    ],
    "T1w_hbp": [
        re.compile(r'.*hbp.*', re.IGNORECASE),
        re.compile(r'.*hepatobiliary.*', re.IGNORECASE),
        re.compile(r'.*20.*min.*', re.IGNORECASE), # Match "20 min"
    ],
    # Add other sequences like T2w, DWI etc. if needed
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_standard_modality_name(series_description: str) -> str:
    """
    Maps a DICOM SeriesDescription to a standard modality name using regex.
    
    Args:
        series_description (str): The SeriesDescription tag from the DICOM header.

    Returns:
        str: The matched standard name, or "unknown_modality" if no match is found.
    """
    if not series_description:
        return "unknown_modality"
        
    for name, patterns in SERIES_MAP.items():
        for pattern in patterns:
            if pattern.search(series_description):
                return name
    return "unknown_modality"

def organize_dicoms_by_series(dicom_root: Path) -> dict:
    """
    Groups all DICOM files in a directory by patient, study, and series.
    
    Args:
        dicom_root (Path): The root directory containing all DICOM files.

    Returns:
        dict: A nested dictionary structured as {patient_id: {study_date: {series_uid: [file_paths]}}}.
    """
    series_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    dicom_files = list(dicom_root.rglob('*.dcm'))
    if not dicom_files:
         dicom_files = list(dicom_root.rglob('*')) # If no .dcm extension, check all files
         dicom_files = [f for f in dicom_files if f.is_file()]


    logging.info(f"Found {len(dicom_files)} potential DICOM files. Start grouping...")
    
    for f_path in tqdm(dicom_files, desc="Grouping DICOMs"):
        try:
            dcm = pydicom.dcmread(f_path, stop_before_pixels=True)
            patient_id = dcm.PatientID
            study_date = dcm.StudyDate
            series_uid = dcm.SeriesInstanceUID
            series_dict[patient_id][study_date][series_uid].append(str(f_path))
        except pydicom.errors.InvalidDicomError:
            # This file is not a valid DICOM, skip it.
            continue
        except Exception as e:
            logging.warning(f"Could not read {f_path}: {e}")

    return series_dict

def convert_and_organize(dicom_root: str, bids_root: str):
    """
    Main function to run the DICOM to BIDS conversion pipeline.
    
    Args:
        dicom_root (str): Path to the source directory with raw DICOMs.
        bids_root (str): Path to the target directory for BIDS-like output.
    """
    dicom_root = Path(dicom_root)
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting DICOM to BIDS conversion.")
    logging.info(f"Source: {dicom_root}")
    logging.info(f"Destination: {bids_root}")

    # 1. Group DICOMs
    grouped_dicoms = organize_dicoms_by_series(dicom_root)
    
    if not grouped_dicoms:
        logging.error("No valid DICOM series found. Please check the source directory.")
        return

    # 2. Convert each series to NIfTI and save in BIDS format
    logging.info("Starting conversion to NIfTI...")
    for patient_id, studies in tqdm(grouped_dicoms.items(), desc="Processing Patients"):
        for study_date, series_collection in studies.items():
            
            # Create BIDS directories
            subject_dir = bids_root / f"sub-{patient_id}"
            session_dir = subject_dir / f"ses-{study_date}"
            anat_dir = session_dir / "anat"
            anat_dir.mkdir(parents=True, exist_ok=True)
            
            for series_uid, file_paths in series_collection.items():
                if not file_paths:
                    continue

                # Use the first file to get series description
                first_file = file_paths[0]
                try:
                    dcm_header = pydicom.dcmread(first_file, stop_before_pixels=True)
                    series_desc = dcm_header.SeriesDescription
                except Exception as e:
                    logging.warning(f"Could not read header for series {series_uid}: {e}")
                    series_desc = ""

                # Map to standard name
                modality_name = get_standard_modality_name(series_desc)
                if modality_name == "unknown_modality":
                    logging.warning(f"Skipping series with unknown description: '{series_desc}' for Patient {patient_id}")
                    continue

                # Define output filename
                output_filename = f"sub-{patient_id}_ses-{study_date}_{modality_name}.nii.gz"
                output_path = anat_dir / output_filename
                
                # Create a temporary directory for this series for conversion
                temp_series_dir = bids_root / "temp_dicom_series"
                temp_series_dir.mkdir(exist_ok=True)
                
                # Symlink or copy files to the temp dir to avoid issues with dicom2nifti
                # (This is a robust way to handle files scattered in different folders)
                for i, f in enumerate(file_paths):
                    os.symlink(f, temp_series_dir / f"{i:04d}.dcm")

                try:
                    # Perform the conversion
                    dicom2nifti.convert_directory(str(temp_series_dir), str(anat_dir), compression=True, reorient=True)
                    
                    # dicom2nifti creates a file with its own name, we need to find and rename it
                    generated_files = list(anat_dir.glob('*.nii.gz'))
                    # Find the newly created file (usually the one that is not our target name yet)
                    # This logic assumes we process one series at a time into the directory
                    # A more robust way might be needed if multiple conversions happen at once
                    # For now, we rename the first file that doesn't match our pattern
                    # Find the file that corresponds to the series we just converted
                    # The safest way is to read the header of the nifti, but that's complex
                    # A simpler but effective way is to find the most recently created file
                    
                    # After conversion, dicom2nifti saves with a default name. We rename it.
                    # Find the file generated by dicom2nifti (it often uses the series number)
                    # and rename it to our BIDS standard.
                    # This is the most brittle part of the process.
                    # A robust solution is to find the most recent file in the directory.
                    
                    # The output file from dicom2nifti can be unpredictable. We find the newest .nii.gz file.
                    list_of_files = anat_dir.glob('*.nii.gz')
                    latest_file = max(list_of_files, key=os.path.getctime)
                    os.rename(latest_file, output_path)

                    logging.info(f"Successfully converted and saved to {output_path}")

                except Exception as e:
                    logging.error(f"Failed to convert series {series_uid} for Patient {patient_id}. Error: {e}")
                finally:
                    # Clean up the temporary directory
                    for f in temp_series_dir.iterdir():
                        f.unlink()
                    temp_series_dir.rmdir()


if __name__ == '__main__':
    # --- Example Usage ---
    # You would run this script from the command line, e.g.:
    # python data_preprocessing/dicom_parser.py --dicom_root /path/to/raw/dicoms --bids_root /path/to/output/bids
    
    import argparse
    parser = argparse.ArgumentParser(description="DICOM to BIDS Conversion and Organization Tool")
    parser.add_argument("--dicom_root", type=str, required=True, help="Root directory containing raw DICOM files.")
    parser.add_argument("--bids_root", type=str, required=True, help="Directory to save the BIDS-structured NIfTI files.")
    
    args = parser.parse_args()
    
    convert_and_organize(args.dicom_root, args.bids_root)