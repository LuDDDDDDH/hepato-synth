# -*- coding: utf-8 -*-
"""
aif_extraction.py

This script automatically extracts the Arterial Input Function (AIF) from a 
4D dynamic contrast-enhanced (DCE) MRI scan.

Core functionalities:
1. Loads a 4D registered and normalized NIfTI image.
2. Automatically locates the abdominal aorta in the arterial phase image based on
   anatomical priors and intensity thresholds.
3. Defines a Region of Interest (ROI) within the aorta.
4. Extracts the mean signal intensity from this ROI across all time points.
5. Saves the resulting time-signal curve (the AIF) to a text file.

A robust AIF is a critical prerequisite for accurate pharmacokinetic modeling
(e.g., DITC model fitting) in the next stage.
"""

import logging
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import argparse
from scipy.ndimage import center_of_mass

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Index of the arterial phase in the 4D image volume.
# Assumes 0:pre, 1:art, 2:pv, 3:eq. This needs to match the data stacking order.
ARTERIAL_PHASE_INDEX = 1
# Expected radius of the aorta in millimeters. Used for creating a search mask.
AORTA_EXPECTED_RADIUS_MM = 15.0
# Intensity percentile to define the brightest region for aorta candidate search.
AORTA_INTENSITY_PERCENTILE = 99.5

def find_aorta_roi(image_4d_sitk: sitk.Image) -> sitk.Image:
    """
    Automatically finds the abdominal aorta and returns a binary mask of its ROI.

    This is a simplified, non-deep-learning approach based on anatomical priors:
    - The aorta is one of the brightest objects in the arterial phase.
    - It is located near the center of the image, anterior to the spine.

    Args:
        image_4d_sitk (sitk.Image): The 4D DCE image (T, Z, Y, X).

    Returns:
        sitk.Image: A 3D binary mask of the aorta ROI, or None if not found.
    """
    # 1. Select the arterial phase volume
    size = list(image_4d_sitk.GetSize())
    if size[3] <= ARTERIAL_PHASE_INDEX:
        logging.error("Arterial phase index is out of bounds for the 4D image.")
        return None
    
    # Extract the 3D arterial volume
    extractor = sitk.ExtractImageFilter()
    size[3] = 0 # We are extracting a 3D volume
    extractor.SetSize(size)
    index = [0, 0, 0, ARTERIAL_PHASE_INDEX]
    extractor.SetIndex(index)
    arterial_img_3d = extractor.Execute(image_4d_sitk)
    
    arterial_arr = sitk.GetArrayFromImage(arterial_img_3d)

    # 2. Threshold the image to find the brightest candidate regions
    threshold_value = np.percentile(arterial_arr[arterial_arr > 0], AORTA_INTENSITY_PERCENTILE)
    binary_mask = sitk.BinaryThreshold(arterial_img_3d, lowerThreshold=float(threshold_value), upperThreshold=arterial_arr.max())
    
    # 3. Find connected components and select the one closest to the image center
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_mask = cc_filter.Execute(binary_mask)
    
    label_shape_stats = sitk.LabelShapeStatisticsImageFilter()
    label_shape_stats.Execute(labeled_mask)
    
    if label_shape_stats.GetNumberOfLabels() == 0:
        logging.warning("Could not find any bright candidate regions for the aorta.")
        return None

    image_center_phys = arterial_img_3d.TransformIndexToPhysicalPoint([s//2 for s in arterial_img_3d.GetSize()])

    min_dist = float('inf')
    best_label = -1
    for label in label_shape_stats.GetLabels():
        # Exclude very small components (noise)
        if label_shape_stats.GetNumberOfPixels(label) < 20:
            continue
        
        centroid_phys = label_shape_stats.GetCentroid(label)
        dist = np.linalg.norm(np.array(image_center_phys) - np.array(centroid_phys))
        
        if dist < min_dist:
            min_dist = dist
            best_label = label
            
    if best_label == -1:
        logging.warning("No suitable aorta candidate component found.")
        return None
    
    # 4. Create the final ROI from the selected component
    aorta_roi = sitk.Equal(labeled_mask, best_label)
    
    # Optional: Erode the mask slightly to avoid partial volume effects at the boundary
    eroder = sitk.BinaryErodeImageFilter()
    eroder.SetKernelRadius(1)
    aorta_roi = eroder.Execute(aorta_roi)

    logging.info(f"Successfully identified aorta ROI with label {best_label}.")
    return aorta_roi


def extract_aif_from_roi(image_4d_sitk: sitk.Image, roi_mask_3d_sitk: sitk.Image) -> np.ndarray:
    """
    Calculates the mean signal intensity within an ROI for each time point.

    Args:
        image_4d_sitk (sitk.Image): The full 4D DCE image.
        roi_mask_3d_sitk (sitk.Image): The 3D binary mask of the aorta.

    Returns:
        np.ndarray: A 2D array of shape (num_time_points, 2), with columns for Time and Signal.
    """
    num_time_points = image_4d_sitk.GetSize()[3]
    
    label_intensity_stats = sitk.LabelIntensityStatisticsImageFilter()
    
    aif_signals = []
    for t in range(num_time_points):
        extractor = sitk.ExtractImageFilter()
        size = list(image_4d_sitk.GetSize())
        size[3] = 0
        extractor.SetSize(size)
        index = [0, 0, 0, t]
        extractor.SetIndex(index)
        volume_3d = extractor.Execute(image_4d_sitk)
        
        # Ensure mask and image have same metadata
        roi_mask_3d_sitk.CopyInformation(volume_3d)

        label_intensity_stats.Execute(roi_mask_3d_sitk, volume_3d)
        
        if 1 in label_intensity_stats.GetLabels():
            mean_signal = label_intensity_stats.GetMean(1)
            aif_signals.append(mean_signal)
        else:
            # This should not happen if ROI is valid
            aif_signals.append(0.0)
            
    # Assuming time points are at 0, 20s, 60s, 180s for example.
    # This should be read from DICOM or a config file in a real scenario.
    time_points_sec = np.array([0, 20, 60, 180, 300]) # Example time points
    
    # Ensure we have the same number of time points
    if len(time_points_sec) != num_time_points:
        logging.warning(f"Number of predefined time points ({len(time_points_sec)}) does not match image time points ({num_time_points}). Using indices instead.")
        time_points_sec = np.arange(num_time_points)
        
    return np.vstack((time_points_sec, np.array(aif_signals))).T


def main(normalized_4d_nifti: str, output_dir: str):
    """
    Main function to run the AIF extraction pipeline for a single 4D image.

    Args:
        normalized_4d_nifti (str): Path to the 4D normalized NIfTI file.
        output_dir (str): Directory to save the output AIF.txt file.
    """
    nifti_path = Path(normalized_4d_nifti)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not nifti_path.exists():
        logging.error(f"Input file not found: {nifti_path}")
        return

    logging.info(f"Loading 4D image: {nifti_path}")
    image_4d = sitk.ReadImage(str(nifti_path), sitk.sitkFloat64)
    
    # 1. Find Aorta ROI
    aorta_roi_mask = find_aorta_roi(image_4d)
    
    if aorta_roi_mask:
        # Save the mask for quality control
        mask_filename = output_dir / nifti_path.name.replace(".nii.gz", "_aorta_mask.nii.gz")
        sitk.WriteImage(aorta_roi_mask, str(mask_filename))
        logging.info(f"Aorta mask saved to {mask_filename}")
        
        # 2. Extract AIF curve
        aif_curve = extract_aif_from_roi(image_4d, aorta_roi_mask)
        
        # 3. Save AIF curve
        aif_filename = output_dir / nifti_path.name.replace(".nii.gz", "_aif.txt")
        np.savetxt(str(aif_filename), aif_curve, fmt="%.4f", header="Time(s) Signal", comments="")
        logging.info(f"AIF curve saved to {aif_filename}")
        
    else:
        logging.error(f"Could not extract AIF for {nifti_path}. Aorta not found.")

if __name__ == '__main__':
    # --- Example Usage ---
    # This would typically be called by a master script that iterates over all patients.
    # python perfusion_modeling/aif_extraction.py \
    #   --input_nifti /path/to/derivatives/normalization/sub-001/ses-date/anat/sub-001_ses-date_4d_stack.nii.gz \
    #   --output_dir /path/to/derivatives/perfusion_modeling/sub-001/ses-date/anat/
    
    parser = argparse.ArgumentParser(description="Automatic Arterial Input Function (AIF) Extraction Tool")
    parser.add_argument("--input_nifti", type=str, required=True, help="Path to the 4D registered and normalized NIfTI file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the AIF curve and mask.")
    
    args = parser.parse_args()
    
    # Note: This script expects a single 4D file. A helper script would be needed
    # to stack the individual 3D phase images into a 4D file first.
    # Or, the script can be modified to accept a directory and stack the images internally.
    
    main(args.input_nifti, args.output_dir)