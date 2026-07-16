import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Load Trained Model
# -----------------------------

model = tf.keras.models.load_model("models/best_model.keras")

# -----------------------------
# Class Names
# -----------------------------

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# -----------------------------
# Prediction Function
# -----------------------------

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224,224))

    image = np.array(image, dtype=np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = prediction[0][predicted_index] * 100

    print("\nPrediction :", class_names[predicted_index])

    print(f"Confidence : {confidence:.2f}%")

    print("\nProbabilities:")

    for i, cls in enumerate(class_names):
        print(f"{cls:10} : {prediction[0][i]*100:.2f}%")

# -----------------------------
# Test
# -----------------------------

predict_image("dataset/glass/glass486.jpg")