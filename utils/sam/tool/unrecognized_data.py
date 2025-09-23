#!/usr/bin/env python3
"""
Scan a directory recursively for PNG files and print paths of those that
raise "unrecognized data stream contents when reading image file" when opened.

Set the directory path directly in the script by modifying ROOT_DIR.
"""
import os
from PIL import Image, UnidentifiedImageError

# === 用户可在此处指定要扫描的根目录 ===
ROOT_DIR = "/mnt/bn/lyl/wwt/anyinsertion/data/mask_prompt"


def scan_bad_pngs(root_dir):
    """
    Walk through `root_dir` and attempt to open each PNG file.
    If opening or verifying the image raises an error containing
    "unrecognized data stream contents", print the file path.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                full_path = os.path.join(dirpath, filename)
                try:
                    with Image.open(full_path) as img:
                        # verify will check for truncated/corrupted data
                        img.verify()
                except (UnidentifiedImageError, OSError) as error:
                    msg = str(error)
                    if 'unrecognized data stream contents' in msg.lower():
                        print(full_path)


def main():
    if not os.path.isdir(ROOT_DIR):
        print(f"Error: '{ROOT_DIR}' is not a valid directory.")
        return

    scan_bad_pngs(ROOT_DIR)

if __name__ == '__main__':
    main()
