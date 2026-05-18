# 🏚️ Building Damage Classification from Satellite Imagery

> A machine learning pipeline for automated building damage assessment from post-disaster satellite imagery — built to support PhD research in disaster response and remote sensing.

---

## 📌 Overview

This project classifies buildings in satellite images into three damage categories following natural disasters (hurricanes, floods, earthquakes). It combines **image segmentation** to extract individual buildings from aerial imagery with **machine learning classification** to assess their damage level.

Built using the **xBD dataset** (Hurricane Harvey post-disaster imagery) with a full data augmentation and balancing pipeline to handle class imbalance.

---

## 🎯 Damage Categories

| Class | Description |
|-------|-------------|
| `major-damage` | Structural collapse or near-complete destruction |
| `minor-damage` | Partial damage, roof damage, flooding |
| `unknown` | Unclear damage level from imagery |

---

## ✨ Pipeline

```
Satellite Image + JSON Annotations
            ↓
Building Segmentation
(crop individual buildings using polygon masks)
            ↓
Data Balancing + Augmentation
(rotation, flip, shear — Albumentations)
            ↓
Train/Test Split (80/20)
            ↓
Feature Extraction
(Color Histograms — HSV space)
            ↓
ML Model Comparison
(Random Forest, SVM, Logistic Regression, XGBoost)
            ↓
Trained Models (.pkl)
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-GPU-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?logo=opencv)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn)

| Component | Technology |
|-----------|-----------|
| Image Processing | OpenCV, PIL |
| Augmentation | Albumentations |
| ML Models | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras (CNN) |
| Feature Extraction | HSV Color Histograms |
| Serialization | Joblib (.pkl models) |

---

## 📁 Project Structure

```
PHD Project/
├── Segmentation Buildings.ipynb   # Extract building crops from satellite images
├── Bulding augmentation.ipynb     # Data balancing & augmentation pipeline
├── NoteBook.ipynb                 # Main ML training & evaluation pipeline
├── CNN.ipynb                      # Deep learning CNN experiments
├── models.ipynb                   # Model comparison experiments
├── Pretrained models.ipynb        # Transfer learning experiments
│
├── major-damage/                  # Raw images + masks
│   ├── images/
│   └── masks/
├── minor-damage/
│   ├── images/
│   └── masks/
├── unknown/
│   ├── images/
│   └── masks/
│
├── Balanced/                      # Augmented balanced dataset
│   ├── major-damage/
│   ├── minor-damage/
│   └── unknown/
│
├── Balanced_Split/                # Train/test split
│   ├── train/
│   ├── val/
│   └── test/
│
├── random_forest.pkl              # Trained Random Forest model
├── svm_rbf.pkl                    # Trained SVM (RBF kernel) model
├── logistic_regression.pkl        # Trained Logistic Regression model
└── xgboost.pkl                    # Trained XGBoost model
```

---

## 🔬 Models Compared

| Model | Notes |
|-------|-------|
| **Random Forest** | 200 estimators, best overall performance |
| **SVM (RBF kernel)** | Strong generalization on small datasets |
| **Logistic Regression** | Baseline linear classifier |
| **XGBoost** | Gradient boosting, handles imbalance well |
| **CNN** | Custom convolutional network (TensorFlow) |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy opencv-python pillow scikit-learn xgboost albumentations tqdm tensorflow joblib matplotlib
```

### Step 1 — Segment Buildings

Run `Segmentation Buildings.ipynb` to extract individual building crops from satellite images and JSON polygon annotations:

```python
process_directory(
    input_dir="path/to/satellite/images",
    output_dir="building_damage_segments",
    output_size=(128, 128),
    pad=5
)
```

### Step 2 — Balance & Augment

Run `Bulding augmentation.ipynb` to apply augmentation and balance the dataset across damage classes.

### Step 3 — Train Models

Run `NoteBook.ipynb` to:
- Load balanced dataset
- Extract HSV color histogram features
- Train and compare all ML models
- Save trained models as `.pkl` files

### Step 4 — Load & Predict

```python
import joblib
import cv2
import numpy as np

# Load model
model = joblib.load("random_forest.pkl")

# Preprocess image
img = cv2.imread("building.png")
img = cv2.resize(img, (128, 128))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
hist = cv2.normalize(hist, hist).flatten().reshape(1, -1)

# Predict
prediction = model.predict(hist)
print(prediction)  # ['major-damage'] / ['minor-damage'] / ['unknown']
```

---

## 📊 Dataset

- **Source:** [xBD Dataset](https://xview2.org/) — xView2 Satellite Imagery
- **Disaster:** Hurricane Harvey (post-disaster imagery)
- **Format:** PNG satellite tiles + JSON polygon annotations
- **Classes:** major-damage, minor-damage, unknown
- **Augmentation:** Rotation (±15°), Horizontal/Vertical Flip, Affine Shear
- **Split:** 80% train / 20% test

---

## 👨‍💻 Author

**Mohamed Osama** — AI Engineer

Developed as a research contribution for a PhD project in disaster response and remote sensing.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/mohamed-osama-558786285)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/MohamedOsama-10)

---

## 📄 License

This project is for academic research purposes.
