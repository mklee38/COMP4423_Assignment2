#!/usr/bin/env python3
# Import standard libraries
import os  # for filesystem operations
import argparse  # for command-line argument parsing
import json  # for saving simple metadata
from datetime import datetime  # for timestamps in logs

# Import numerical and image libraries
import numpy as np  # for numeric arrays and operations
import pandas as pd  # for tabular logging and saving
from skimage.io import imread  # to read images from disk
from skimage.transform import resize  # to resize images to fixed shape
from skimage.color import rgb2gray  # to convert images to grayscale for HOG
from skimage.feature import hog  # to compute HOG features

# Import scikit-learn for modelling and utilities
from sklearn.ensemble import RandomForestClassifier  # the classifier we will train
from sklearn.preprocessing import LabelEncoder  # to convert string labels to integers
import joblib  # to save and load model objects


def list_image_files(root_dir):
    """Return a list of (filepath, label) for images under root_dir arranged by class subfolders."""
    items = []  # accumulator for (path, label) tuples
    # iterate sorted class directories for deterministic ordering
    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)  # full path to potential class folder
        if not os.path.isdir(cls_path):
            continue  # skip non-directories
        # iterate files in class folder
        for fname in sorted(os.listdir(cls_path)):
            lower = fname.lower()  # lowercase name for extension checks
            if lower.endswith(('.jpg', '.jpeg', '.png')):
                items.append((os.path.join(cls_path, fname), cls))  # store path and class
    return items  # return collected list


def extract_feature(image_path, resize_shape=(128, 128), hog_pixels_per_cell=(16, 16)):
    """Extract a simple feature vector (HOG + color histogram) from an image file."""
    img = imread(image_path)  # read image as numpy array
    # if grayscale image, convert to 3-channel by stacking
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)  # make pseudo-RGB for histogram
    # resize image to fixed shape for consistent features
    img_resized = resize(img, resize_shape, anti_aliasing=True, preserve_range=True)
    img_resized = img_resized.astype(np.uint8)  # ensure integer pixel range
    # compute HOG on grayscale representation
    gray = rgb2gray(img_resized)  # convert to grayscale for HOG
    hog_feat = hog(gray, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2), feature_vector=True)
    # compute per-channel color histograms and concatenate
    chans = []  # accumulator for channel histograms
    for c in range(img_resized.shape[2]):
        hist, _ = np.histogram(img_resized[:, :, c], bins=32, range=(0, 255))  # compute histogram
        hist = hist.astype(float)  # convert to float before normalization
        hist /= (hist.sum() + 1e-8)  # normalize histogram to sum to 1
        chans.append(hist)  # append channel histogram
    color_feat = np.concatenate(chans)  # concatenate channel histograms
    feat = np.concatenate([hog_feat, color_feat])  # combine HOG and color features
    return feat  # return 1-D feature vector


def main():
    parser = argparse.ArgumentParser(description='Train a classical classifier on the vegetation dataset')
    parser.add_argument('--train-dir', default='data/train', help='Path to training data root')
    parser.add_argument('--model-out', default='models/model.pkl', help='Where to save trained model')
    parser.add_argument('--log-out', default='logs/train_log.json', help='Where to save training log json')
    args = parser.parse_args()  # parse CLI arguments

    # collect image paths and labels from train directory
    items = list_image_files(args.train_dir)  # list of (path, label)
    if not items:
        raise SystemExit(f'No images found in {args.train_dir}. Please populate dataset following Spec.md')

    # extract features for all images and collect labels
    features = []  # list to hold feature arrays
    labels = []  # list to hold label strings
    for path, label in items:
        feat = extract_feature(path)  # compute feature vector for image
        features.append(feat)  # add to list
        labels.append(label)  # add label

    X = np.stack(features, axis=0)  # shape (N, D)
    y = np.array(labels)  # convert labels to numpy array

    # encode labels to integers for classifier
    le = LabelEncoder()  # create label encoder
    y_enc = le.fit_transform(y)  # fit and transform label strings

    # create and fit classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # ensemble classifier
    clf.fit(X, y_enc)  # train model on features and encoded labels

    # ensure output directories exist
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)  # create models/ if missing
    os.makedirs(os.path.dirname(args.log_out), exist_ok=True)  # create logs/ if missing

    # save the model and label encoder together using joblib
    joblib.dump({'model': clf, 'label_encoder': le}, args.model_out)  # persist model and encoder

    # prepare a small training log with stats and timestamps
    log = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'train_dir': args.train_dir,
        'n_samples': int(X.shape[0]),
        'feature_dim': int(X.shape[1]),
        'classes': list(le.classes_),
        'model_path': args.model_out,
    }

    # write training log as json for quick inspection
    with open(args.log_out, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)  # write pretty JSON

    # also save a CSV summary using pandas for human-friendly viewing
    df = pd.DataFrame({'filepath': [p for p, _ in items], 'label': labels})  # create DataFrame of samples
    csv_path = os.path.join(os.path.dirname(args.log_out), 'train_samples.csv')  # path for sample CSV
    df.to_csv(csv_path, index=False)  # save CSV without index

    print('Training complete. Model saved to', args.model_out)  # user feedback
    print('Training log saved to', args.log_out)  # user feedback


if __name__ == '__main__':
    main()  # run the training pipeline when executed as a script
