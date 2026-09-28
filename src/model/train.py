import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from src.model.model import build_model
from src.utils.config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    MODEL_DIR,
    MODEL_PATH
)


def train_model():

    # ==============================
    # Load Training Dataset
    # ==============================

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["buffalo", "cattle"],
        shuffle=True,
        seed=42
    )

    # ==============================
    # Load Validation Dataset
    # ==============================

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["buffalo", "cattle"],
        shuffle=False
    )

    # ==============================
    # Calculate Class Weights
    # ==============================

    # Buffalo = 0
    # Cattle = 1

    class_counts = np.array([1125, 4954])

    classes = np.array([0, 1])

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.repeat(classes, class_counts)
    )

    class_weights = {
        0: class_weights_array[0],
        1: class_weights_array[1]
    }

    print("\nClass Weights:")
    print(f"Buffalo: {class_weights[0]:.4f}")
    print(f"Cattle: {class_weights[1]:.4f}")

    # ==============================
    # Improve Data Loading
    # ==============================

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    validation_dataset = validation_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    # ==============================
    # Build Model
    # ==============================

    model = build_model()

    # Create model directory
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==============================
    # Callbacks
    # ==============================

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )

    # ==============================
    # Train Model
    # ==============================

    print("\nStarting model training...\n")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[
            early_stopping,
            reduce_lr,
            model_checkpoint
        ]
    )

    print("\nTraining completed!")
    print(f"Best model saved at: {MODEL_PATH}")

    return history


if __name__ == "__main__":
    train_model()