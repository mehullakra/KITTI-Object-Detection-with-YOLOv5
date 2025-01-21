import os
import random
import shutil
from tqdm import tqdm

def split_dataset(image_dir, label_dir, output_dir, split_ratio = 0.8):
    """
    Splits the dataset into training and validation sets.

    Args:
        image_dir (str): directory containing images.
        label_dir (str): directory containing YOLO label files.
        output_dir (str): Base directory for output dataset structure.
        split_ratio (float): ratio of dataset for training (default 0.8).
    """

    # Paths for train and validation split
    train_image_dir = os.path.join(output_dir, "images/train")
    val_image_dir = os.path.join(output_dir, "images/val")
    train_label_dir = os.path.join(output_dir, "labels/train")
    val_label_dir = os.path.join(output_dir, "labels/val")

    for path in [train_image_dir, val_image_dir, train_label_dir, val_label_dir]:
        os.makedirs(path, exist_ok=True)

    # fetch all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    # shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # move files to respective directories
    for image_file in tqdm(train_files, desc="Processing training files"):
        base_name = os.path.splitext(image_file)[0]
        shutil.move(os.path.join(image_dir, image_file), os.path.join(train_image_dir, image_file))
        label_file = os.path.join(label_dir, f"{base_name}.txt")
        if os.path.exists(label_file):
            shutil.move(label_file, os.path.join(train_label_dir, f"{base_name}.txt"))

    for image_file in tqdm(val_files, desc="Processing validation files"):
        base_name = os.path.splitext(image_file)[0]
        shutil.move(os.path.join(image_dir, image_file), os.path.join(val_image_dir, image_file))
        label_file = os.path.join(label_dir, f"{base_name}.txt")
        if os.path.exists(label_file):
            shutil.move(label_file, os.path.join(val_label_dir, f"{base_name}.txt"))

    print("Dataset split completed")