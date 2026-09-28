from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parents[2]

# Dataset Paths
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# Image Settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Training Settings
EPOCHS = 10

# Classes
CLASS_NAMES = ["buffalo", "cattle"]

# Model Path
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "animal_classifier.keras"