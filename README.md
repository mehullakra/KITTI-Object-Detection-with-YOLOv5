# Object Detection with YOLOv5 on KITTI Dataset

## Step 0: Create Conda Environment

Create and activate the Conda environment:

```bash
conda create -n kittiyolo
conda activate kittiyolo
```

Install necessary dependencies:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install cuda -c nvidia/label/cuda-12.4.0
```

## Step 1: Install YOLOv5

Clone the YOLOv5 repository and install its dependencies:

in `custom_requirements.txt` we comment out the pytorch related code since we already installed it.

```bash
git clone https://github.com/ultralytics/yolov5
pip install -r custom_requirements.txt
```

## Step 2: Prepare the Dataset

1. Download the KITTI dataset from the KITTI Vision Benchmark Suite - https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
   
   Specifically, download:

   - left color images: ```data_object_image_2.zip```

   - training labels of object data set: ```data_object_label_2.zip```

2. Place the downloaded files inside the data folder.

3. Extract the files to get the folders ```training``` and ```testing```.

4. The ```testing``` folder is not needed for this implementation; you can delete it or keep it.

## Step 3: Run Data Preprocessing Scripts

These scripts convert the input data format to YOLO-compatible format, where the bounding box (bbox) labels are represented by:
- `class_id`, `x_center(normalized)`, `y_center(normalized)`, `width(normalized)`, `height(normalized)`

This replaces the KITTI dataset format of `x_min`, `y_min`, `x_max`, `y_max`. Normalization ensures bbox coordinates remain accurate regardless of image resizing.

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

To train the model, we will use the YOLOv5s model with its pretrained weights, to finetune on our dataset. Pretrained weights are auto-downloaded. The `data.yaml` file specifies information about the dataset, including:
- `train path`
- `val path`
- `Number of classes`

Run the following command:

```bash
python yolov5/train.py --img 640 --epochs 30 --data data.yaml --weights yolov5s.pt
```

according to best practices - https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/#training-settings

you should use `300 epochs` as a starting point, but for the sake of this experiment and limited resources we choose arbitrary number - `30`

training results are saved in `yolov5\runs\train` where you will find training metrics as well as the best and last weights for the model

note we use `image size` as 640 because that is what the input of the yolov5s model is

## Step 5: Validate the best model

from the project root directory run the following command
```bash
python yolov5/val.py --weights yolov5\runs\train\exp7\weights\best.pt --data data.yaml --img 640
```

the results will be saved to `yolov5\runs\val\exp`

## Step x: Quantize Model and then Export to TensorRT then deploy on edge device and upload demo
