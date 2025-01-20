import os

def initialize_for_yolo(kitti_label_path, yolo_label_path, image_dir, output_dir, split_ratio = 0.8):
    """
    initializes the kitti dataset for yolo training by converting and splitting

    Args:
        kitti_label_path (str): Path to KITTI annotations.
        yolo_label_path (str): Path to save converted YOLO labels.
        image_dir (str): Path to KITTI images.
        output_dir (str): Path for YOLO training dataset structure.
        split_ratio (float): Fraction of data to use for training (default 0.8).
    """

    # step 1: run convert_to_yolo.py to convert annotations from kitti to yolo format
    print("Converting KITTI annotations to YOLO format...")
    from scripts.convert_to_yolo import convert_to_yolo
    convert_to_yolo(kitti_label_path, yolo_label_path, image_dir)

    # step 2: run split_dataset.py to convert folder structure to yolo format
    print("Splitting dataset into train and val...")
    from scripts.split_dataset import split_dataset
    split_dataset(image_dir, yolo_label_path, output_dir, split_ratio)

    print("Initialization completed.")

if __name__ == "__main__":
    # define paths based on the current folder structure
    kitti_label_path = "data/training/label_2/"
    yolo_label_path = "data/training/yolo_labels/"
    image_dir = "data/training/image_2/"
    output_dir = "datasets/kitti/"

    # Initialize the dataset for YOLO
    initialize_for_yolo(kitti_label_path, yolo_label_path, image_dir, output_dir)