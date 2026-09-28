import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape=(224, 224, 3)):
    """
    Build a cattle vs buffalo image classification model
    using MobileNetV2 transfer learning.
    """

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # Load pretrained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Build model
    inputs = layers.Input(shape=input_shape)

    x = data_augmentation(inputs)

    # Convert pixel values from [0,255] to [-1,1]
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1
    )(x)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs
    )

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model