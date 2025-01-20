import os
import cv2

def get_image_dimensions(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    return width, height

def convert_to_yolo(kitti_label_path, yolo_label_path, image_dir):
    # Mapping classes to class_id
    class_mapping = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

    # Ensuring the yolo directory exists
    os.makedirs(yolo_label_path, exist_ok=True)

    # Iterate over all KITTI label files
    for label_file in os.listdir(kitti_label_path):
        # Get corresponding image file
        base_name = os.path.splitext(label_file)[0]
        image_file = os.path.join(image_dir, f"{base_name}.png") # since they are .png files

        # Skip if image does not exist
        if not os.path.exists(image_file):
            print(f"Image file {image_file} not found. Skipping...")
            continue

        # Get the image dimensions
        width, height = get_image_dimensions(image_file)

        # Paths for KITTI and YOLO Labels
        kitti_file = os.path.join(kitti_label_path, label_file)
        yolo_file = os.path.join(yolo_label_path, label_file)

        with open(kitti_file, "r") as kitti_f, open(yolo_file, "w") as yolo_f:
            for line in kitti_f:
                data = line.strip().split()

                # Parse the fields
                class_name = data[0]
                x1, y1, x2, y2 = map(float, data[4:8])

                # Skip if the class not in class_mapping
                if class_name not in class_mapping:
                    continue

                # convert bbox to yolo format
                class_id = class_mapping[class_name]
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                # Write to YOLO format
                yolo_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")