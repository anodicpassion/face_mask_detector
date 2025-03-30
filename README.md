# 😷 Face Mask Detection using Deep Learning

A real-time face mask detection system built with TensorFlow, Keras, OpenCV, and MobileNetV2. This project can detect whether a person is wearing a face mask or not using a webcam feed.

---

## 📌 Features

- Real-time face detection with OpenCV
- MobileNetV2-based CNN model for mask classification
- Transfer learning for high accuracy with limited training
- Data augmentation for improved generalization
- Easy deployment via Flask and Docker (optional)

---

## 📂 Project Structure
```.
├── LICENSE
├── data
│   ├── download.bash
│   └── download_data_manually.txt
├── evaluate.py
├── file_structure
├── haarCascade
│   ├── __init__.py
│   └── haarcascade_frontalface_default.xml
├── model
│   └── mobilenet_v2.h5
├── preprocess.py
├── requirements.txt 
├── test_real_time.py
└── train.py

4 directories, 11 files
```
---

## 🧠 Model

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Input Shape**: 224x224x3
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anodicpassion/face_mask_detector.git
cd face_mask_detector
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv            # On Windows: python -m venv venv 
source venv/bin/activate        # On Windows: venv\Scripts\activate
```


### 3. Install Dependencies 

```bash
pip install -r requirements.txt
```

### 4. Download data

```bash
bash data/download.bash         # On Windows: read download_data_manually.txt
```

## 🚀 Running the Project


### 1. Train the Model

```bash
python3 train.py
```
The trained model will be saved to: `model/mask_detector_model.h5`

### 2. Run Real-Time Face Mask Detection

```bash
python3 test_real_time.py
```
Press q to exit the webcam window.





