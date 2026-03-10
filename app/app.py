#!/usr/bin/env python3
# Simple CLI application to run single-image predictions using the trained model
import argparse  # parse command-line arguments
import os  # filesystem utilities
import joblib  # load saved model
import numpy as np  # numeric arrays
from skimage.io import imread  # read image files
from skimage.transform import resize  # resize images
from skimage.color import rgb2gray  # grayscale conversion for HOG
from skimage.feature import hog  # HOG feature extraction


def extract_feature(image_path, resize_shape=(128, 128), hog_pixels_per_cell=(16, 16)):
    """Extract features for a single image in the same way as training script."""
    img = imread(image_path)  # read image
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)  # convert grayscale to 3-channel
    img_resized = resize(img, resize_shape, anti_aliasing=True, preserve_range=True)  # resize
    img_resized = img_resized.astype(np.uint8)  # ensure uint8 pixels
    gray = rgb2gray(img_resized)  # convert to grayscale for HOG
    hog_feat = hog(gray, pixels_per_cell=hog_pixels_per_cell, cells_per_block=(2, 2), feature_vector=True)  # compute HOG
    chans = []  # accumulate per-channel histograms
    for c in range(img_resized.shape[2]):
        hist, _ = np.histogram(img_resized[:, :, c], bins=32, range=(0, 255))  # histogram
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-8)  # normalize
        chans.append(hist)
    color_feat = np.concatenate(chans)  # concatenate channel histograms
    feat = np.concatenate([hog_feat, color_feat])  # combine into single vector
    return feat  # return feature vector


def main():
    parser = argparse.ArgumentParser(description='Predict class for a single image using trained model')
    parser.add_argument('image', help='Path to image file to predict')
    parser.add_argument('--model', default='models/model.pkl', help='Path to trained model')
    args = parser.parse_args()  # parse CLI args

    if not os.path.exists(args.image):
        raise SystemExit(f'Image not found: {args.image}')  # exit if input missing
    if not os.path.exists(args.model):
        raise SystemExit(f'Model not found: {args.model}. Run scripts/train.py first')  # exit if model missing

    obj = joblib.load(args.model)  # load saved model and label encoder
    clf = obj['model']  # classifier object
    le = obj['label_encoder']  # label encoder

    feat = extract_feature(args.image)  # compute features for the input image
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba([feat])[0]  # get probability vector
        top_idx = int(probs.argmax())  # index of top prediction
        top_conf = float(probs[top_idx])  # confidence of top prediction
    else:
        top_idx = int(clf.predict([feat])[0])  # predicted index without probabilities
        top_conf = 1.0  # fallback confidence

    pred_label = le.inverse_transform([top_idx])[0]  # convert index back to human label
    print(f'Predicted: {pred_label} (confidence: {top_conf:.3f})')  # print result for user


if __name__ == '__main__':
    main()  # run CLI app when script is executed
