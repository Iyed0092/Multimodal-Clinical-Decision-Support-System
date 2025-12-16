import os
import shutil
import random

SOURCE_DIR = "backend/data/raw/brats_mri"  
TRAIN_DIR = "backend/data/raw/brats_train"
TEST_DIR = "backend/data/raw/brats_test"
SPLIT_RATIO = 0.8  


def split_dataset():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: source directory not found: {SOURCE_DIR}")
        return

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    patients = [p for p in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, p))]
    total = len(patients)

    if total == 0:
        print("Error: no patient folders found.")
        return

    print(f"Found {total} patient folders. Shuffling and splitting...")

    random.seed(42)
    random.shuffle(patients)

    split_idx = int(total * SPLIT_RATIO)
    train_patients = patients[:split_idx]
    test_patients = patients[split_idx:]

    print(f"Moving {len(train_patients)} -> TRAIN, {len(test_patients)} -> TEST")

    for p in train_patients:
        src = os.path.join(SOURCE_DIR, p)
        dst = os.path.join(TRAIN_DIR, p)
        shutil.move(src, dst)

    for p in test_patients:
        src = os.path.join(SOURCE_DIR, p)
        dst = os.path.join(TEST_DIR, p)
        shutil.move(src, dst)


    print("Split complete.")
    print(f"Train directory: {TRAIN_DIR}")
    print(f"Test directory : {TEST_DIR}")


if __name__ == "__main__":
    split_dataset()
