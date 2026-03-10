#!/usr/bin/env python3
# Standard imports
import os  # for filesystem manipulation
import argparse  # to parse command-line args
import csv  # to write CSV results

# Numeric and image libraries
import numpy as np  # numerical arrays
from skimage.io import imread  # read images
from skimage.transform import resize  # resize to fixed size
from skimage.color import rgb2gray  # convert to grayscale
from skimage.feature import hog  # HOG feature extraction

# ML utilities
import joblib  # load saved model and encoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # evaluation metrics
import matplotlib.pyplot as plt  # for confusion matrix visualization


def list_image_files(root_dir):
    """Collect (filepath, label) pairs from the given root directory structured by class subfolders."""
    items = []  # accumulator
    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)  # full path for class folder
        if not os.path.isdir(cls_path):
            continue  # skip files
        for fname in sorted(os.listdir(cls_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append((os.path.join(cls_path, fname), cls))  # add file and label
    return items  # return collected list


def extract_feature(image_path, resize_shape=(128, 128), hog_pixels_per_cell=(16, 16)):
    """Extract the same feature vector used during training (HOG + color hist)."""
    img = imread(image_path)  # read image
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)  # convert grayscale to 3-channel
    img_resized = resize(img, resize_shape, anti_aliasing=True, preserve_range=True)  # resize
    img_resized = img_resized.astype(np.uint8)  # ensure integer pixels
    gray = rgb2gray(img_resized)  # grayscale for HOG
    hog_feat = hog(gray, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2), feature_vector=True)  # HOG
    chans = []  # accumulate color histograms
    for c in range(img_resized.shape[2]):
        hist, _ = np.histogram(img_resized[:, :, c], bins=32, range=(0, 255))  # histogram per channel
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-8)  # normalize
        chans.append(hist)
    color_feat = np.concatenate(chans)  # concatenate channel histograms
    feat = np.concatenate([hog_feat, color_feat])  # combine features
    return feat  # return vector


def plot_and_save_confusion(cm, classes, out_path):
    """Plot a confusion matrix and save to disk."""
    fig, ax = plt.subplots(figsize=(8, 6))  # create figure and axes
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # display matrix
    ax.figure.colorbar(im, ax=ax)  # add colorbar
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)), xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label')  # label axes
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')  # rotate x labels
    thresh = cm.max() / 2.  # threshold for text color
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')  # write numbers
    fig.tight_layout()  # adjust layout
    fig.savefig(out_path)  # save figure to file
    plt.close(fig)  # close figure to free memory


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on test dataset')
    parser.add_argument('--test-dir', default='data/test', help='Path to test data root')
    parser.add_argument('--model', default='models/model.pkl', help='Path to saved model')
    parser.add_argument('--out-dir', default='logs', help='Directory to save evaluation outputs')
    args = parser.parse_args()  # parse CLI arguments

    # load model and label encoder
    obj = joblib.load(args.model)  # load saved dict with keys 'model' and 'label_encoder'
    clf = obj['model']  # classifier object
    le = obj['label_encoder']  # label encoder

    # collect test items
    items = list_image_files(args.test_dir)  # list of (path, label)
    if not items:
        raise SystemExit(f'No test images found in {args.test_dir}. Please populate dataset following Spec.md')

    # prepare output directory
    os.makedirs(args.out_dir, exist_ok=True)  # create output directory if missing

    # iterate and predict
    rows = []  # rows for CSV output
    y_true = []  # true labels (encoded)
    y_pred = []  # predicted labels (encoded)
    for path, label in items:
        feat = extract_feature(path)  # compute feature
        probs = None  # default when classifier has no predict_proba
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba([feat])[0]  # get probability vector
            pred_idx = int(probs.argmax())  # index of top class
            conf = float(probs[pred_idx])  # confidence of top class
        else:
            pred_idx = int(clf.predict([feat])[0])  # predicted index
            conf = 1.0  # fallback confidence
        pred_label = le.inverse_transform([pred_idx])[0]  # convert index back to string label
        rows.append({'filepath': path, 'true_label': label, 'pred_label': pred_label, 'confidence': conf})  # append row
        y_true.append(label)  # store true
        y_pred.append(pred_label)  # store predicted

    # save results CSV
    csv_path = os.path.join(args.out_dir, 'results.csv')  # path to results CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filepath', 'true_label', 'pred_label', 'confidence'])  # CSV writer
        writer.writeheader()  # write header
        for r in rows:
            writer.writerow(r)  # write each result row

    # compute metrics and save a text summary
    acc = accuracy_score(y_true, y_pred)  # overall accuracy
    report = classification_report(y_true, y_pred, digits=4)  # detailed per-class report
    cm = confusion_matrix(y_true, y_pred, labels=list(le.classes_))  # confusion matrix with consistent ordering
    metrics_path = os.path.join(args.out_dir, 'metrics.txt')  # path for metrics summary
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f'Accuracy: {acc:.4f}\n\n')  # write accuracy
        f.write('Classification report:\n')  # header
        f.write(report)  # write full report

    # save confusion matrix visualization
    cm_path = os.path.join(args.out_dir, 'confusion_matrix.png')  # path for confusion matrix image
    plot_and_save_confusion(cm, list(le.classes_), cm_path)  # create and save plot

    print('Evaluation complete.')  # user feedback
    print('Results saved to', csv_path)  # user feedback
    print('Metrics saved to', metrics_path)  # user feedback
    print('Confusion matrix saved to', cm_path)  # user feedback


if __name__ == '__main__':
    main()  # run evaluation when executed directly
