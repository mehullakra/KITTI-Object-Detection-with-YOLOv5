# Multi-Object-Detection-with-YOLO-on-KITTI-Dataset

## Step 0: Create Conda Environment

Create and activate the Conda environment:

```bash
conda create -n kittiyolo python=3.8 -y
conda activate kittiyolo
```

Install necessary dependencies:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install cuda -c nvidia/label/cuda-11.8.0
```

## Step 1: Install YOLOv5

Clone the YOLOv5 repository and install its dependencies:

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt
```

## Step 2: Prepare the Dataset

1. Download the KITTI dataset from the KITTI Vision Benchmark Suite - https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
   Specifically, download:

```
left color images: data_object_image_2.zip
training labels of object data set: data_object_label_2.zip
```

2. Place the downloaded files ```(data_object_image_2.zip and data_object_label_2.zip)``` inside the data folder.

3. Extract the files to get the folders ```training``` and ```testing```.

4. The ```testing``` folder is not needed for this implementation; you can delete it or keep it.

## Step 3: Run Data Preprocessing Scripts

These scripts convert the input data format to YOLO-compatible format, where the bounding box (bbox) labels are represented by:
- `class_id, x_center(normalized), y_center(normalized), width(normalized), height(normalized)`

This replaces the KITTI dataset format of `x_min, y_min, x_max, y_max`. Normalization ensures bbox coordinates remain accurate regardless of image resizing.

Run the commands below from the project root directory:

```bash
python -m scripts.initialize_for_yolo
```

This will create a new folder called `datasets` in the project root directory. The files will move from `data/training/` to `datasets/kitti/`.

### Step 3.1 (Optional): Visualize and Verify Converted Annotations

To ensure the converted annotations and images align correctly, run:

```bash
python -m scripts.visualize_data
```

## Step 4: Model Training

To train the model, we will use the YOLOv5s model with its pretrained weights, finetuned on our dataset. Pretrained weights are auto-downloaded. The `data.yaml` file specifies information about the dataset, including:
- `train` path
- `val` path
- Number of classes

Run the following command:

```bash
python yolov5/train.py --img 640 --epochs 30 --data data.yaml --weights yolov5s.pt
```

All training results are saved to `yolov5/runs/train/`, including:
- Best weights
- Last weights
- Useful training and performance metrics
