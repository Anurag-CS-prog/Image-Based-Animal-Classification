import random
import shutil
from pathlib import Path

from src.utils.config import BASE_DIR


# ==============================
# Dataset Configuration
# ==============================

SOURCE_DIR = BASE_DIR / "dataset" / "source" / "clean_images"
OUTPUT_DIR = BASE_DIR / "dataset"

CLASSES = ["buffalo", "cattle"]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ==============================
# Get Images Recursively
# ==============================

def get_images(class_name):
    class_dir = SOURCE_DIR / class_name

    images = [
        file
        for file in class_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return images


# ==============================
# Copy Images
# ==============================

def copy_images(images, destination, class_name):

    destination.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(images, start=1):

        # Create unique filename
        new_name = f"{class_name}_{index:05d}{image_path.suffix.lower()}"

        destination_path = destination / new_name

        shutil.copy2(image_path, destination_path)


# ==============================
# Prepare Dataset
# ==============================

def prepare_dataset():

    random.seed(RANDOM_SEED)

    for class_name in CLASSES:

        print(f"\nProcessing: {class_name.upper()}")

        images = get_images(class_name)

        print(f"Total images found: {len(images)}")

        # Shuffle images
        random.shuffle(images)

        total = len(images)

        train_end = int(total * TRAIN_RATIO)

        validation_end = train_end + int(
            total * VALIDATION_RATIO
        )

        train_images = images[:train_end]

        validation_images = images[
            train_end:validation_end
        ]

        test_images = images[
            validation_end:
        ]

        print(f"Train: {len(train_images)}")
        print(f"Validation: {len(validation_images)}")
        print(f"Test: {len(test_images)}")

        # Copy images
        copy_images(
            train_images,
            OUTPUT_DIR / "train" / class_name,
            class_name
        )

        copy_images(
            validation_images,
            OUTPUT_DIR / "validation" / class_name,
            class_name
        )

        copy_images(
            test_images,
            OUTPUT_DIR / "test" / class_name,
            class_name
        )

        print(f"{class_name.upper()} completed.")


# ==============================
# Main
# ==============================

if __name__ == "__main__":

    print("=" * 50)
    print("Preparing Animal Classification Dataset")
    print("=" * 50)

    prepare_dataset()

    print("\nDataset preparation completed successfully!")