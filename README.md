# Deepfake Detection using SVM and GLCM Features

## Overview

This project implements a deepfake image detection system using classical machine learning techniques. It uses GLCM (Gray Level Co-occurrence Matrix) for texture-based feature extraction and Support Vector Machine (SVM) for classification.

The objective is to classify images as real or fake based on statistical texture patterns present in facial images.

---

## Environment

Python Version: 3.11

---

## Methodology

### Feature Extraction (GLCM)

* Images converted to grayscale
* Resized to 128 × 128
* Extracted texture features:

  * Contrast
  * Energy
  * Homogeneity
  * Correlation
* Multiple distances and angles used for better representation

---

### Preprocessing

* Feature normalization using StandardScaler
* Dataset split into training and testing sets

---

### Model

* Algorithm: Support Vector Machine (SVM)
* Kernel: RBF
* Hyperparameter tuning using RandomizedSearchCV

**Best Parameters:**

* C = 10
* gamma = scale
* kernel = rbf

---

## Results

* Validation Accuracy: 63.10%
* Test Accuracy: 61.36%

### Classification Performance

**Class 0 (Real):**

* Precision: 0.64
* Recall: 0.53
* F1-score: 0.58

**Class 1 (Fake):**

* Precision: 0.60
* Recall: 0.70
* F1-score: 0.64

**Overall Accuracy:** 61%

**Dataset Size Used:**

* Training: 30,000 images
* Testing: 10,000 images

---

## Observations

* GLCM captures texture inconsistencies in deepfake images
* Model performs better at detecting fake images (higher recall)
* Classical ML provides a lightweight alternative to deep learning
* Performance is moderate due to limited feature complexity

---

## Project Structure

```
passion-project-deep-fake-identifiers/
│
├── src/
│   └── utils/
│       └── train_svm_glcm.py
│
├── models/
├── deepfake_dataset/
├── reports/
├── notebooks/
│
├── README.md
├── requirements.txt
```

---

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the model:

```
python src/utils/train_svm_glcm.py
```

---

## Dataset

Dataset is not included due to large size.

### Expected structure:

```
deepfake_dataset/
   real-vs-fake/
      train/
         real/
         fake/
      test/
         real/
         fake/
```

---

## Author

Liyakat Hussain

---

## Note

This project demonstrates a baseline deepfake detection system using classical machine learning techniques.


---

# 🚀 Improved Approach: MLP with LBP, GLCM & FFT

The following section presents an enhanced deepfake detection pipeline using multiple feature extraction techniques and a neural network-based classifier.

---






# 🧠 Deepfake Detection using LBP, GLCM & FFT + MLP

## 📌 Overview

This project implements a **classical machine learning pipeline** for deepfake detection using handcrafted features:

* **LBP (Local Binary Patterns)** → texture features
* **GLCM (Gray Level Co-occurrence Matrix)** → spatial relationships
* **FFT (Fast Fourier Transform)** → frequency artifacts

These features are used to train **MLP (Multi-Layer Perceptron)** models for classification.

---

## 📂 Dataset

* Dataset: Real vs Fake Images
* Size: **~140,000 images**
* Structure:

```
deepfake_dataset/
    real-vs-fake/
        train/
        valid/
        test/
```

---

## ⚙️ Feature Extraction

| Feature      | Description        | Dimension |
| ------------ | ------------------ | --------- |
| LBP          | Texture patterns   | 128       |
| GLCM         | Spatial statistics | 24        |
| FFT          | Frequency domain   | 15        |
| **Combined** | All features       | 167       |

---

## 🚀 Models Used

* MLP Classifier (scikit-learn)
* Separate models trained on:

  * LBP
  * GLCM
  * FFT
  * Combined Features

---

## 📊 Results

### 🔹 GLCM Model

* Accuracy: **0.6665**
* Precision: ~0.67
* Recall: ~0.67

### 🔹 LBP Model

* Accuracy: **0.7993**
* Precision: ~0.80
* Recall: ~0.80

### 🔹 FFT Model

* Accuracy: **0.6295**
* Precision: ~0.63
* Recall: ~0.63

### 🔥 Combined Model (Best)

* Accuracy: **0.8235**
* Precision: ~0.82
* Recall: ~0.82
* F1-score: ~0.82

---

## 📈 Key Insights

* LBP performs best among individual features
* FFT alone is weak but adds value in combination
* Combining features improves performance significantly
* Classical ML can achieve strong results without CNNs

---

## 🏗️ Project Structure

```
src/
│
├── utils/
│   ├── feature_extractor.py
│   ├── lbp_features.py
│   ├── glcm_features.py
│   ├── fft_features.py
│   └── combined_features/
│
├── models/
│   ├── *.pkl (trained models)
│
├── pipeline/
│   └── trainer.py
│
├── train_mlp.py
└── main.py
```

---

## 💾 Outputs

After training:

* Trained models:

  * `lbp_mlp.pkl`
  * `glcm_mlp.pkl`
  * `fft_mlp.pkl`
  * `combined_mlp.pkl`

* Stored in:

```
src/models/
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train models

```bash
python -m src.train_mlp
```

---

## ⚠️ Notes

* Dataset is **not included** in repo (too large)
* `.pkl` files are ignored via `.gitignore`
* Designed for CPU-based training (no GPU required)

---

## 🔮 Future Work

* Replace MLP with CNN / Vision Transformers
* Add real-time deepfake detection
* Improve FFT feature engineering
* Hyperparameter tuning

---

## 👨‍💻 Author

Liyakat Hussain

---
