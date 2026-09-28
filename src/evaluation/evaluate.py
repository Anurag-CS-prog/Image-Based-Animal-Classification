import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from src.utils.config import (
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    MODEL_PATH
)


def evaluate_model():
    # Load test dataset
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["buffalo", "cattle"],
        shuffle=False
    )

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Get predictions
    predictions = model.predict(test_dataset, verbose=0)

    # Convert probabilities to class labels
    predicted_labels = (predictions >= 0.5).astype(int).flatten()

    # Get actual labels
    actual_labels = np.concatenate([
        labels.numpy()
        for _, labels in test_dataset
    ]).astype(int)

    # Accuracy
    accuracy = accuracy_score(actual_labels, predicted_labels)

    print("\nTest Accuracy:", f"{accuracy * 100:.2f}%")

    # Classification Report
    print("\nClassification Report:")
    print(
        classification_report(
            actual_labels,
            predicted_labels,
            target_names=["Buffalo", "Cattle"]
        )
    )

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(actual_labels, predicted_labels))


if __name__ == "__main__":
    evaluate_model()