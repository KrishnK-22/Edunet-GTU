import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_dataset():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        "dataset",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        "dataset",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = (
        train_dataset
        .cache()
        .shuffle(1000)
        .prefetch(buffer_size=AUTOTUNE)
    )

    validation_dataset = (
        validation_dataset
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(x), y)
    )

    validation_dataset = validation_dataset.map(
    lambda x, y: (preprocess_input(x), y)
    )
    return train_dataset, validation_dataset, class_names