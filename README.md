# ROP-Project
Traditional ML baselines for ROP classification using retinal fundus images. Includes HOG+SVM and Sobel+SVM pipelines for binary and multi-class setups. This repo is the first phase of a larger project focused on building deep learning models to automatically grade ROP stages from clinical retinal images.
**Supervisor:** Dr. Shyam Rajagopalan  
**Author:** Vedavalli V  
**Duration:** April – July 2025  

---

## 📌 Project Overview

This project implements **classical machine learning pipelines** for automated detection of **Retinopathy of Prematurity (ROP)** from retinal fundus images.

Instead of deep learning, this implementation focuses on:

- Feature engineering (HOG, Sobel, Color Histograms)
- Patient-level aggregation
- SVM-based classification
- Strict prevention of data leakage

Three models were developed:

1. **HOG + Color Histogram + SVM (Binary)**
2. **Sobel Edge Features + SVM (Binary)**
3. **Sobel Edge Features + SVM (Multi-Class)**

---

## Table of Contents

- [Dataset Structure](#dataset-structure)
- [Methods](#methods)
  - [1️ HOG + SVM (Binary)](#1️⃣-hog--svm-binary)
  - [2️ Sobel + SVM (Binary)](#2️⃣-sobel--svm-binary)
  - [3️ Sobel + SVM (Multi-Class)](#3️⃣-sobel--svm-multi-class)
- [Pipeline Workflow](#pipeline-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Design Decisions](#design-decisions)
- [Future Improvements](#future-improvements)

---

# Dataset Structure

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


Each folder corresponds to a **single patient/sample**.

Metadata CSV must contain:

| Column | Description |
|--------|------------|
| ID | Patient/sample ID |
| Diagnosis Code | ROP stage |

---

# Methods

---

## 1 HOG + SVM (Binary)

###  Task
Classify:
ROP vs Normal


###  Features
- Histogram of Oriented Gradients (HOG)
- RGB Color Histograms
- CLAHE contrast enhancement
- Circular masking

###  Because
Captures:
- Vessel orientation
- Retinal texture
- Color distribution

---

## 2️ Sobel + SVM (Binary)

### Task
Classify: ROP vs Normal


### Features
From Sobel gradient magnitude:

- Mean edge strength
- Standard deviation
- Maximum magnitude
- Histogram of gradient distribution

### Why?
ROP progression affects **vascular tortuosity**, which is better captured via edge gradients.

---

## 3 Sobel + SVM (Multi-Class)

### Task
Classify into: ROP
Immature Retina
Other



### 🎯 Task
Classify:
