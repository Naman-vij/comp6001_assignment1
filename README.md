# Image Restoration and Object Detection Analysis

## Overview
This project investigates the impact of image deblurring on object detection performance. 
Blurred images are restored using classical deblurring techniques, and their influence on 
detection accuracy is analyzed using a YOLO-based object detection model.

The project covers:
- Image deblurring (Wiener and Richardson–Lucy)
- Object detection on blurred and restored images
- Training a detection model on deblurred data
- Performance comparison across blurred, deblurred, and sharp images

---

## Project Structure
comp6001_assignment1/
│
├── notebooks/ # Jupyter notebooks for all tasks
├── dataset_v2/ # Training dataset
│ ├── images/train/
│ ├── labels/train/
│ └── data.yaml
│
├── data/
│ ├── blur/ # Blurred images
│ ├── sharp/ # Original sharp images
│
├── results/ # Deblurred and detection outputs
├── runs/ # YOLO training outputs (auto-generated)
│
└── README.md


---

## Methods

### Image Deblurring
- Wiener Filter  
- Richardson–Lucy Deconvolution  

### Object Detection
- YOLOv8 (Ultralytics)

### Evaluation Metrics
- PSNR (Peak Signal-to-Noise Ratio)  
- SSIM (Structural Similarity Index)  
- Detection count and confidence scores  

---

## How to Run

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python scikit-image matplotlib pandas

2. Run Notebooks
## assignment1.ipynb

. Train Model
from ultralytics import YOLO

model = YOLO("yolov10n.pt")
model.train(
    data="../dataset_v2/data.yaml",
    epochs=40,
    imgsz=1024,
    batch=4,
    device=0 to activate gpu
)
