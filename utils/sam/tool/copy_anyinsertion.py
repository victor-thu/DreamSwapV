import os
import shutil

# Source and destination directories
tar_dir = '/mnt/bn/lyl/wwt/anyinsertion/data/mask_prompt/train/accessory/tar_image'
dst_dir = '/mnt/bn/lyl/wwt/Wan2.1/i2v_img'

# Ensure destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# List of image indices to copy
indices = [
    32, 37, 46, 54, 61, 64, 78, 80, 89, 92,
    1092, 3016, 3018, 3030, 3043, 3089, 3100
]

for idx in indices:
    filename = f"train_accessory_{idx}.png"
    src_path = os.path.join(tar_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    if os.path.isfile(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {filename}")
    else:
        print(f"Warning: Source file does not exist: {filename}")

print("Done copying selected images.")
