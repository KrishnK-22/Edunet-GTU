import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/best_model.keras")

model = load_model()

# -----------------------------
# Class Information
# -----------------------------

waste_info = {
    "cardboard": {
        "bin": "🟦 Blue Recycling Bin",
        "tip": "Flatten cardboard boxes before recycling.",
        "impact": "Recycling cardboard saves trees and reduces landfill waste."
    },

    "glass": {
        "bin": "🟩 Green Glass Bin",
        "tip": "Remove caps before recycling.",
        "impact": "Glass can be recycled endlessly without losing quality."
    },

    "metal": {
        "bin": "🟨 Metal Recycling Bin",
        "tip": "Rinse food containers before recycling.",
        "impact": "Recycling metal saves huge amounts of energy."
    },

    "paper": {
        "bin": "🟦 Paper Recycling Bin",
        "tip": "Keep paper dry and clean.",
        "impact": "Paper recycling reduces deforestation."
    },

    "plastic": {
        "bin": "🟦 Plastic Recycling Bin",
        "tip": "Wash plastic bottles before recycling.",
        "impact": "Plastic recycling reduces ocean pollution."
    },

    "trash": {
        "bin": "⬛ General Waste Bin",
        "tip": "Dispose responsibly.",
        "impact": "Proper disposal prevents environmental contamination."
    }
}

class_names = list(waste_info.keys())

# -----------------------------
# Title
# -----------------------------

st.title("♻️ EcoSort AI")

st.write("AI Powered Waste Classification System")

# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg","jpeg","png"]
)

# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224,224))

    img = np.array(img).astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)

    confidence = prediction[0][index] * 100

    label = class_names[index]

    info = waste_info[label]

    st.success(f"Prediction : {label.upper()}")

    st.info(f"Confidence : {confidence:.2f}%")

    st.subheader("♻️ Disposal Bin")

    st.write(info["bin"])

    st.subheader("🌱 Recycling Tip")

    st.write(info["tip"])

    st.subheader("🌍 Environmental Impact")

    st.write(info["impact"])

    st.subheader("🎯 Sustainable Development Goals")

    st.markdown("""
- ✅ SDG 12 : Responsible Consumption and Production
- ✅ SDG 13 : Climate Action
""")