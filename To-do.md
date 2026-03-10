# To-Do List — Campus Vegetation Recognition Assignment

## Task 1 — Class definition & data collection (5 marks)

- Select at least 8 vegetation classes (trees, shrubs, flowers, grasses, etc.).
- Write a collection plan: locations, viewpoints, lighting, time of day.
- Ensure intra-class variation (angles, distances, occlusions, backgrounds).

## Task 2 — Dataset building & labeling (10 marks)

- Collect images using on-site name plates where possible for ground truth.
- Capture multiple images per plant across different angles/lighting/locations.
- Organize dataset: `data/train/<class>/`, `data/val/<class>/`, `data/test/<class>/`.
- Produce an image count table per class and perform sanity checks (duplicates, corrupted files).
- Include sample photos per class in the report.

## Task 3 — Train a classifier (10 marks)

- Implement `scripts/train.py` using `scikit-learn`.
- Load the dataset, extract handcrafted features (HOG, color histograms, LBP, or combos), and train classical classifiers (SVM, Random Forest, Gradient Boosting).
- Perform model selection (validation set or cross-validation) and set random seeds for reproducibility.
- Save trained model (e.g., `models/model.pkl`) and training logs.

## Task 4 — Evaluation & error analysis (5 marks)

- Implement `scripts/evaluate.py` to run the saved model on the test set.
- Report accuracy, confusion matrix, and per-class precision/recall/F1 (macro-F1).
- Save `results.csv` (filepath, true_label, pred_label[, confidence]) and artifacts: `metrics.txt`, `confusion_matrix.png`, example folders.
- Create qualitative folders for correct/wrong examples and analyze failure modes.

## Task 5 — Application (20 marks)

- Provide a simple local app (e.g., `app/app.py` using Streamlit/Flask) to upload/select an image and show predicted class.
- Include launch and usage instructions in `README.md`.

## Task 6 — Report (50 marks)

- Use the provided template and include all required details, experiments, and an AI usage section.
- Name the report: `Assignment2_Report_<ID>_<name>.pdf`.

## Bonus (up to 10 marks)

- Extra credit for high-quality code, documentation, reproducibility, and strong results.

## General

- No deep learning usage allowed for this assignment.
- Prepare submission ZIP: `Assignment2_Code_<ID>_<name>.zip`.
- Upload deadline: 11:59 PM, 24 March 2026.
- Avoid plagiarism: all work must be original.