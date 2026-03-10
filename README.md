# Campus Vegetation Recognition Project

## Overview

This repository contains an assignment for COMP4423 (Computer Vision) to build an end-to-end image classification pipeline for recognizing at least eight categories of campus vegetation using a self-collected dataset. The pipeline uses traditional machine learning (no deep learning) and includes data collection, training, evaluation, and a simple application for image prediction.

### Key components

- **Dataset:** Self-collected images of campus plants with variations in viewpoint, lighting, background, etc.
- **Model:** Traditional ML classifiers (e.g., SVM, Random Forest, Gradient Boosting) with handcrafted features (HOG, color histograms, LBP).
- **Application:** Local runnable app to upload/select an image and predict the plant class.
- **Report:** Document methods, results, and any AI assistance.

## Directory structure

```
project_root/
├── data/                  # Dataset folders
│   ├── train/
│   │   └── <class_name>/  # e.g., oak_tree/*.jpg
│   ├── val/
│   │   └── <class_name>/
│   └── test/
│       └── <class_name>/
├── examples/              # For error analysis (correct/wrong predictions)
├── models/                # Saved models (e.g., model.pkl)
├── logs/                  # Training logs, metrics
├── app/                   # Application code (e.g., app.py)
├── scripts/               # Training and evaluation scripts
│   ├── train.py
│   └── evaluate.py
├── report/                # Report files and images
├── README.md              # This file
├── To-do.md               # Task list
└── Spec.md                # Detailed specifications
```

## Installation

1. Clone or set up the repository.
2. Install dependencies (Python 3.12+ recommended):

```bash
pip install scikit-learn scikit-image numpy pandas matplotlib joblib
```

> Note: No deep learning libraries (TensorFlow/PyTorch) should be used for this assignment.

For the simple application UI consider using `streamlit` or `flask`.

## Usage

- **Data collection:** Organize images into `data/train/`, `data/val/`, `data/test/` by class.
- **Training:**

```bash
python scripts/train.py
```

Outputs: `models/model.pkl`, `logs/train_log.txt`, `logs/val_metrics.json`.

- **Evaluation:**

```bash
python scripts/evaluate.py
```

Outputs: `logs/metrics.txt`, `logs/confusion_matrix.png`, `logs/results.csv`, and example image folders.

- **Application:**

```bash
python app/app.py
```

Follow on-screen instructions to upload/select an image and get predictions.

## Notes

- Ensure reproducibility: set random seeds in code.
- If generative AI is used for assistance (e.g., code snippets), verify and document usage and limitations in the report.
- Submission: prepare a report PDF and a ZIP of code/datasets as specified in `Spec.md`.