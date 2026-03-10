#!/usr/bin/env python3
"""Create train/val/test splits from a single labeled-folder dataset.

Usage example:
    python scripts/split_dataset.py --source labeled_dataset --out data --train 0.7 --val 0.15 --test 0.15 --seed 42

This script copies files (does not move) from the source folder structure:
    labeled_dataset/<class>/*.jpg
into the destination structure:
    data/train/<class>/...
    data/val/<class>/...
    data/test/<class>/...

Each line in the file contains a short comment describing its purpose.
"""

import os  # filesystem utilities
import argparse  # command-line parsing
import random  # deterministic shuffling using seed
import shutil  # copying files
from glob import glob  # convenient file listing by pattern


def is_image_file(filename):
    """Return True if filename has an image extension we recognize."""
    lower = filename.lower()  # normalize case for extension check
    return lower.endswith(('.jpg', '.jpeg', '.png'))  # accepted extensions


def collect_class_files(source_dir):
    """Return a dict mapping class_name -> list of file paths found under source_dir/class_name."""
    classes = {}  # accumulator for class -> file list
    for entry in sorted(os.listdir(source_dir)):
        class_path = os.path.join(source_dir, entry)  # full path to candidate class folder
        if not os.path.isdir(class_path):
            continue  # skip non-folders at top level
        # gather image files under this class folder
        files = [os.path.join(class_path, f) for f in sorted(os.listdir(class_path)) if is_image_file(f)]
        if files:
            classes[entry] = files  # only include classes with at least one image
    return classes  # return mapping


def make_dirs_for_split(out_root, split_names, class_names):
    """Create output directories for each split and class (e.g., data/train/<class>)."""
    for split in split_names:
        for cls in class_names:
            path = os.path.join(out_root, split, cls)  # e.g., data/train/oak
            os.makedirs(path, exist_ok=True)  # create if missing


def copy_file_list(file_list, dest_dir):
    """Copy files in file_list to dest_dir preserving filenames."""
    for src in file_list:
        dst = os.path.join(dest_dir, os.path.basename(src))  # destination path
        shutil.copy2(src, dst)  # copy file and metadata


def split_and_copy(source_dir, out_root, train_ratio, val_ratio, test_ratio, seed=42, copy=True):
    """Split each class's files by the provided ratios and copy them into out_root splits."""
    classes = collect_class_files(source_dir)  # get mapping class -> files
    if not classes:
        raise SystemExit(f'No class folders with images found in {source_dir}')  # exit if nothing found

    # weights list and split names for convenience
    splits = ['train', 'val', 'test']
    ratios = [train_ratio, val_ratio, test_ratio]  # expected to sum to 1.0

    # create output directories per class and split
    make_dirs_for_split(out_root, splits, classes.keys())

    rand = random.Random(seed)  # deterministic randomness using seed

    for cls, files in classes.items():
        files_copy = list(files)  # make a shallow copy we can shuffle
        rand.shuffle(files_copy)  # shuffle in place
        n = len(files_copy)  # how many samples for this class

        # compute split indices (floor for train/val, remainder to test)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val  # whatever remains goes to test

        # slice lists for each split
        train_files = files_copy[:n_train]
        val_files = files_copy[n_train:n_train + n_val]
        test_files = files_copy[n_train + n_val:]

        # copy files into their target class-specific folders
        if copy:
            if train_files:
                copy_file_list(train_files, os.path.join(out_root, 'train', cls))
            if val_files:
                copy_file_list(val_files, os.path.join(out_root, 'val', cls))
            if test_files:
                copy_file_list(test_files, os.path.join(out_root, 'test', cls))

    # return a short summary for logging or inspection
    return {'n_classes': len(classes), 'seed': seed}


def parse_args():
    """Parse CLI arguments for dataset splitter."""
    p = argparse.ArgumentParser(description='Create train/val/test splits from a labeled-folder dataset')
    p.add_argument('--source', required=True, help='Source folder with class subfolders (labeled dataset)')
    p.add_argument('--out', default='data', help='Output root where train/val/test folders will be created')
    p.add_argument('--train', type=float, default=0.7, help='Train split ratio (default 0.7)')
    p.add_argument('--val', type=float, default=0.15, help='Validation split ratio (default 0.15)')
    p.add_argument('--test', type=float, default=0.15, help='Test split ratio (default 0.15)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    p.add_argument('--move', action='store_true', help='Move files instead of copying (destructive)')
    return p.parse_args()  # return parsed args


def main():
    args = parse_args()  # parse CLI args

    # validate ratios roughly sum to 1.0 (allow small floating point slack)
    s = args.train + args.val + args.test  # sum of ratios
    if abs(s - 1.0) > 1e-6:
        raise SystemExit('Train/val/test ratios must sum to 1.0')  # exit with helpful message

    # create splits (copy by default; move if --move given)
    result = split_and_copy(args.source, args.out, args.train, args.val, args.test, seed=args.seed, copy=not args.move)

    print('Split complete:', result)  # brief summary printed to user
    print(f"Created folders under {args.out}/train, {args.out}/val, {args.out}/test")  # guide user where to look


if __name__ == '__main__':
    main()  # execute when script is run directly
