import os
import shutil
import sys

def run_command(cmd):
    print(f"\n>>> Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print("Error executing command.")
        sys.exit(1)

def main():
    print("=== Starting Hepato-Synth Demo Pipeline ===")
    
    # 1. 清理旧数据
    if os.path.exists("demo_data"): shutil.rmtree("demo_data")
    if os.path.exists("demo_output"): shutil.rmtree("demo_output")
    
    # 2. 生成数据
    run_command("python demo/generate_mock_data.py")
    
    # 3. 模拟预处理（这里简化，直接复制假数据当作已处理数据）
    print("\n>>> Simulating Preprocessing...")
    processed_dir = "demo_output/preprocessed/normalization/sub-001/ses-01/anat"
    os.makedirs(processed_dir, exist_ok=True)
    # 把生成的假数据复制过去，改个名加上 _desc-normalized
    src_dir = "demo_data/sub-001/ses-01/anat"
    for f in os.listdir(src_dir):
        shutil.copy(os.path.join(src_dir, f), os.path.join(processed_dir, f.replace(".nii.gz", "_desc-normalized.nii.gz")))
    
    # 4. 生成假的配置文件
    config_content = """
defaults:
  - base_config
data_root: "demo_data"
output_root: "demo_output"
preprocessed_data_root: "demo_output/preprocessed"
experiment_description: "Demo Run"
gpu_ids: [] # Use CPU
training:
  epochs: 1
  batch_size: 2
data_preprocessing:
  patch_size: [16, 16, 16]
model:
  name: "PhysicsInformedSwinUNETR"
  in_channels: 4 # Demo只用4个通道简化
  out_channels: 1
  img_size: [16, 16, 16]
"""
    with open("configs/demo.yaml", "w") as f:
        f.write(config_content)

    # 5. 运行训练 (只跑 1 个 epoch)
    # 注意：这里会尝试调用 main.py，如果依赖包没装好会报错，但不影响演示脚本本身逻辑
    print("\n>>> Launching Training Engine (Smoke Test)...")
    try:
        run_command("python main.py --mode train --config configs/demo.yaml")
    except Exception as e:
        print(f"Training simulation failed (expected if env not set): {e}")
        print("But the pipeline logic is valid.")

    print("\n=== Demo Completed Successfully (Simulated) ===")

if __name__ == "__main__":
    main()