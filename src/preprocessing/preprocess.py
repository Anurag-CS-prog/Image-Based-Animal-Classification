from PIL import Image
import numpy as np

from src.utils.config import IMAGE_SIZE


def preprocess_image(image_path):
    """
    Load and preprocess an animal image.

    Steps:
    1. Load image
    2. Convert to RGB
    3. Resize image
    4. Convert to NumPy array
    5. Add batch dimension

    Note:
    MobileNetV2 preprocessing is already included
    inside the trained model.
    """

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array