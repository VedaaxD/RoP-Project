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

```bash
pip install numpy pandas scikit-learn opencv-python pillow matplotlib tqdm joblib scikit-image
