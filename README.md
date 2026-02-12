# ROP-Project

Traditional ML baselines for ROP classification using retinal fundus images.  
Includes HOG+SVM and Sobel+SVM pipelines for binary and multi-class setups.  

This repository represents the first phase of a larger project focused on building deep learning models to automatically grade ROP stages from clinical retinal images.

Supervisor: Dr. Shyam Rajagopalan  
Author: Vedavalli V  
Duration: April – July 2025  

---

## Project Overview

This project implements classical machine learning pipelines for automated detection of Retinopathy of Prematurity (ROP) from retinal fundus images.

Instead of deep learning, this phase focuses on:

- Feature engineering (HOG, Sobel, Color Histograms)
- Patient-level aggregation
- SVM-based classification
- Strict prevention of patient-level data leakage

Three baseline models were developed:

1. HOG + Color Histogram + SVM (Binary)
2. Sobel Edge Features + SVM (Binary)
3. Sobel Edge Features + SVM (Multi-Class)

---

## Table of Contents

- [Dataset Structure](#dataset-structure)
- [Methods](#methods)
  - [HOG + SVM (Binary)](#hog--svm-binary)
  - [Sobel + SVM (Binary)](#sobel--svm-binary)
  - [Sobel + SVM (Multi-Class)](#sobel--svm-multi-class)
- [Pipeline Workflow](#pipeline-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Design Decisions](#design-decisions)
- [Future Improvements](#future-improvements)

---

## Dataset Structure

Expected directory format:

base_dir/
│
├── 001/
│ ├── img1.jpg
│ ├── img2.jpg
│
├── 002/
│ ├── img1.jpg
│
└── ...


Each folder corresponds to a single patient/sample.

Metadata CSV must contain:

| Column | Description |
|--------|------------|
| ID | Patient/sample ID |
| Diagnosis Code | ROP stage or status |

---

## Methods

### HOG + SVM (Binary)

**Task:**  
Classify ROP vs Normal.

**Features:**

- Histogram of Oriented Gradients (HOG)
- RGB color histograms
- CLAHE contrast enhancement
- Circular masking to remove black borders

**Rationale:**  
HOG captures structural and directional vessel patterns, while color histograms capture retinal intensity distribution.

---

### Sobel + SVM (Binary)

**Task:**  
Classify ROP vs Normal.

**Features extracted from Sobel gradient magnitude:**

- Mean edge strength
- Standard deviation of edge strength
- Maximum gradient magnitude
- Histogram of gradient distribution

**Rationale:**  
ROP progression alters vascular tortuosity, which is better captured using edge-based gradient features.

---

### Sobel + SVM (Multi-Class)

**Task:**  
Classify into three categories:

- ROP
- Immature Retina
- Other

**Label Mapping:**

| Diagnosis Code | Class |
|----------------|-------|
| 1 | Immature |
| 2–8 | ROP |
| Others | Other |

This is a disease-state classifier, not stage-wise severity classification.

---

## Pipeline Workflow

1. Read metadata CSV  
2. Collect images grouped by sample folder  
3. Preprocess each image:
   - Resize to 224 × 224  
   - Apply CLAHE  
   - Apply circular mask  
4. Extract features:
   - HOG + color histogram OR
   - Sobel gradient statistics  
5. Aggregate image-level features into a single sample-level feature vector (mean pooling)  
6. Split data using GroupShuffleSplit (patient-level separation)  
7. Standardize features using StandardScaler  
8. Perform hyperparameter tuning using GridSearchCV  
9. Train SVM classifier (RBF kernel)  
10. Evaluate using:
    - Accuracy
    - Classification report
    - Confusion matrix  
11. Save trained model and visual outputs  

---

## Installation

Install required packages:

bash
pip install numpy pandas scikit-learn opencv-python pillow matplotlib tqdm joblib scikit-image



## Usage

Each script supports two modes: `train` and `predict`.

### Training a Model


python <script_name>.py --mode train \
--base_dir /path/to/images \
--csv metadata.csv

During training, the pipeline performs:

Metadata loading and validation
Image collection grouped by patient
Image preprocessing (resize, CLAHE, circular mask)
Feature extraction (HOG or Sobel)
Sample-level feature aggregation
Patient-level train/test split using GroupShuffleSplit
Feature scaling using StandardScaler
Hyperparameter tuning with GridSearchCV
SVM training (RBF kernel)
Evaluation and artifact generation

### Outputs

After training, the following artifacts are generated:
| File            | Description            |
| --------------- | ---------------------- |
| `model.joblib`  | Trained SVM classifier |
| `scaler.joblib` | Fitted feature scaler  |

### Sample-Level Predictions
test_predictions_sample_level.csv contains:
| Column    | Description        |
| --------- | ------------------ |
| sample_id | Patient folder ID  |
| id_int    | Numeric patient ID |
| label     | Ground truth label |
| pred      | Predicted label    |


### Future Improvements
1. Feature Fusion
Combine: HOG, Sobel, Color histograms, Vessel enhancement filters (e.g., Frangi)
To improve discriminative power.

2. Stage-Wise ROP Classification
Extend from grouped "ROP" to:
Stage 1
Stage 2
Stage 3
Stage 4
Stage 5
AP-ROP
For finer clinical grading.

3. Group-Based Cross-Validation
Implement K-fold group cross-validation to:

Improve statistical reliability
Report confidence intervals

4. Deep Learning Baselines

Next project phase includes:
Transfer learning (ResNet, EfficientNet)
End-to-end CNN training
Grad-CAM explainability

To compare classical ML with deep learning approaches.

