import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
NUM_CLASSES = 6


def build_model():

    # Load MobileNetV2 without its original classifier
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze the pretrained layers
    base_model.trainable = False

    # Build our model
    model = models.Sequential([

        layers.Input(shape=(224, 224, 3)),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),

        layers.Dropout(0.4),

        layers.Dense(NUM_CLASSES, activation="softmax")

    ])

    return model