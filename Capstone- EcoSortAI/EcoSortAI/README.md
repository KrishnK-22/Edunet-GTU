# ♻️ EcoSort AI

An AI-powered waste classification system that identifies waste into six categories using a deep learning model based on MobileNetV2. The application helps users correctly segregate waste, promoting recycling and environmental sustainability.

---

## 🌍 Project Overview

EcoSort AI is developed as part of the **AI with Green Skills Internship**. The application uses Artificial Intelligence to classify waste images and provides recycling guidance, encouraging sustainable waste management practices.

---

## ✨ Features

- 📷 Upload an image of waste
- 🤖 AI-powered waste classification
- 📊 Confidence score for predictions
- ♻️ Recycling recommendation
- 🌱 Environmental awareness tips
- 💻 Simple and interactive Streamlit web application

---

## 🧠 AI Model

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Framework:** TensorFlow / Keras
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy

### Supported Waste Categories

- 📦 Cardboard
- 🍾 Glass
- 🥫 Metal
- 📄 Paper
- 🧴 Plastic
- 🗑️ Trash

---

## 📊 Model Performance

| Metric | Value |
|--------|------:|
| Training Accuracy | **93.5%** |
| Validation Accuracy | **85.7%** |

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Streamlit
- NumPy
- Pandas
- Plotly
- Matplotlib
- Pillow

---

## 📂 Project Structure

```text
EcoSortAI/
│
├── app.py
├── train.py
├── predict.py
├── dataset.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
│
├── models/
│   └── best_model.keras
│
├── outputs/
│   ├── accuracy.png
│   └── loss.png
│
├── assets/
│
└── dataset/
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/EcoSortAI.git
```

Go to the project folder:

```bash
cd EcoSortAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 Future Enhancements

- 📷 Live camera support
- 🌍 Multi-language interface
- 📍 Nearby recycling center locator
- 📱 Mobile-friendly version
- ☁️ Cloud deployment

---

## 👨‍💻 Developer

**Krishn Khatri**

AI with Green Skills Internship Project

---

## 📜 License

This project is developed for educational and internship purposes.