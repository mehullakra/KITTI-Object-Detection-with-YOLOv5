# Multi-Object-Detection-with-YOLO

step 0: create conda environment
conda create -n kittiyolo python=3.8 -y
conda activate kittiyolo
install a few dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install cuda -c nvidia/label/cuda-11.8.0
pip install opencv-python


step 1. install yolov5 in your project directory 
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt

step 2. create folder called "data" in the project directory, and place the files "data_object_image_2.zip" and "data_object_label_2.zip"
then extract them.
now you will have the folders - "training" and "testing". we don't need testing for this implementation so you can either delete it or keep it.

step 3. run the data preprocessing scripts - these are important since we need to convert the input data format to yolo compatible format, where the bbox labels are represented by class_id, x_center(normalized), y_center(normalized), width(normalized), height(normalized). instead of x_min, y_min, x_max, y_max like the kitti dataset format. this normalization is good because no matter what you resize the image to the bbox coordinates are always accurate.

run commands below from project root directory

python -m scripts.initialize_for_yolo

now you will have a new folder called datasets in the project root directory. and your files would have moved from data/training/ to dataset/kitti/

step 3.1(optional). visualize and verify the converted annotations and the images align by running

python -m scripts.visualize_data

step 4. model training
to train the model we will use the yolov5s model and use it's pretrained weights to finetune on our dataset.
Pretrained weights are auto-downloaded
data.yaml contains information about our dataset that the model will be trained on.
it specifies the train path, val path, and number of classes.

python yolov5/train.py --img 640 --epochs 30 --data data.yaml --weights yolov5s.pt

All training results are saved to yolov5/runs/train/
that is also where the best weights and the last weights can be found. as well as other useful training and performance metrics
