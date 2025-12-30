import os
import numpy as np
import nibabel as nib
from pathlib import Path

def create_mock_nifti(shape, path):
    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, path)

def main():
    root = Path("demo_data")
    # 模拟 BIDS 结构
    anat_dir = root / "sub-001" / "ses-01" / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating mock data in {anat_dir}...")
    
    # 生成 4 个期相
    phases = ["T1w_precontrast", "T1w_arterial", "T1w_portal", "T1w_equilibrium"]
    for p in phases:
        create_mock_nifti((32, 32, 32), anat_dir / f"sub-001_ses-01_{p}.nii.gz")
    
    # 生成假掩膜
    mask_data = np.zeros((32, 32, 32), dtype=np.uint8)
    mask_data[10:20, 10:20, 10:20] = 1 # Liver
    img = nib.Nifti1Image(mask_data, np.eye(4))
    nib.save(img, "demo_liver_mask.nii.gz") # 放在根目录方便测试

if __name__ == "__main__":
    main()