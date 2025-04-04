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
├── requirements.txt 
├── preprocess.py
├── test_real_time.py
└── train.py

4 directories, 11 files
```
