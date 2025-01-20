import os
import cv2

def visualize_annotation(image_dir, label_dir, class_names, num_samples = 5):
    """visualizes bounding boxes from YOLO annotations on images"""
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)

    for i, image_file in enumerate(image_files[:num_samples]):
        # make sure corresponding label exists
        base_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(label_dir, f"{base_name}.txt")
        if not os.path.exists(label_file):
            continue

        # read image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        # draw the bboxes
        with open(label_file, "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x_center, y_center, width, height = (
                    x_center * image.shape[1],
                    y_center * image.shape[0],
                    width * image.shape[1],
                    height * image.shape[0],
                )
                # converting to x_min, x_max, y_min, y_max for visualizing with opencv
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)

                # draw rectangle and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, class_names[int(class_id)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 255, 0), 1)

        # display image
        cv2.imshow("Annotation", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # define paths
    image_dir = "datasets/kitti/images/train"
    label_dir = "datasets/kitti/labels/train"
    class_names = ["Car", "Pedestrian", "Cyclist"]

    # visualize annotations
    visualize_annotation(image_dir, label_dir, class_names)