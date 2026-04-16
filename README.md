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
