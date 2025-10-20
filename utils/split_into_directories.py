import os
import shutil
import random


def split_json_files(source_dir, train_dir, test_dir, train_ratio=0.7, seed=42):
    """
    Split the JSON files in the source directory into train and test directories
    """
    random.seed(seed)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]

    random.shuffle(json_files)
    split_index = int(len(json_files) * train_ratio)

    train_files = json_files[:split_index]
    test_files = json_files[split_index:]

    for f in train_files:
        shutil.move(os.path.join(source_dir, f), os.path.join(train_dir, f))

    for f in test_files:
        shutil.move(os.path.join(source_dir, f), os.path.join(test_dir, f))

    print(f"Moved {len(train_files)} files to {train_dir}")
    print(f"Moved {len(test_files)} files to {test_dir}")


source_directory = '../data'
train_directory = '../data/train_convs'
test_directory = '../data/test_convs'

split_json_files(source_directory, train_directory, test_directory)