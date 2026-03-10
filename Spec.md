# Specifications for Campus Vegetation Recognition Project

## General requirements

- **Dataset:** Self-collected, at least 8 classes, images in JPG format.
- Cover real-world variations (backgrounds, lighting, viewpoints, angles, distances, occlusions).
- Use plant name plates for ground-truth labeling.
- Folder structure: `data/train/<class_name>/*.jpg`, `data/val/<class_name>/*.jpg`, `data/test/<class_name>/*.jpg`.

- **ML pipeline:** Traditional ML only (no neural networks / deep learning).
- **Libraries:** `scikit-learn`, `scikit-image` (for feature extraction).
- **Features:** Handcrafted (HOG, color histograms, LBP, or combinations).
- **Classifiers:** SVM, Random Forest, Gradient Boosting, etc.

- **Reproducibility:** Set random seeds and provide runnable code.
- **AI collaboration:** Allowed but must be documented and verified in the report.
- **Submission naming:**

```text
Assignment2_Report_<student_ID>_<name>.pdf
Assignment2_Code_<student_ID>_<name>.zip
```

No plagiarism. Late penalty: 10% per day (up to 3 days, then zero).

## Tasks and marks

### Task 1: Class definition & data collection plan (5 marks)

- Define ≥ 8 classes.
- Provide a detailed plan (locations, viewpoints, lighting, time) covering intra-class variation.

### Task 2: Dataset building & labeling (10 marks)

- Build and label dataset using on-site name plates when possible.
- Include: sample photos per class, image count table, and sanity checks (duplicates, corrupted files).

### Task 3: Train a classifier (10 marks)

- Implement a training script to load data, extract handcrafted features, and train classical classifiers.
- Perform model selection (validation set or cross-validation) and justify choices.
- Save outputs: trained model, training log (features, hyperparameters, validation performance).

### Task 4: Evaluation & error analysis (5 marks)

- Evaluate on the test set and report accuracy, confusion matrix, and per-class metrics.
- Produce qualitative examples (correct/incorrect) and analyze failure modes.
- Outputs: `results.csv`, `metrics.txt`, `confusion_matrix.png`, example image folders.

### Task 5: Application (20 marks)

- Provide a simple local app (e.g., `app.py` with Streamlit/Flask) for image upload and prediction.
- Include launch and usage instructions.

### Task 6: Report (50 marks)

- Complete report covering methods, experiments, results, and AI usage documentation.

### Bonus (up to 10 marks)

- Extra credit for code quality, documentation, high performance, and well-presented report.