import tensorflow as tf
import matplotlib.pyplot as plt
import os

from dataset import load_dataset
from model import build_model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------
# Load Dataset
# -------------------------

train_ds, val_ds, class_names = load_dataset()

print("\nClasses:")
print(class_names)

# -------------------------
# Build Model
# -------------------------

model = build_model()

# -------------------------
# Compile Model
# -------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

# -------------------------
# Create folders
# -------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -------------------------
# Callbacks
# -------------------------

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)

checkpoint = ModelCheckpoint(

    "models/best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

# -------------------------
# Train
# -------------------------

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=15,

    callbacks=[early_stop, checkpoint]

)

# -------------------------
# Save Model
# -------------------------

model.save("models/final_model.keras")

print("\nModel Saved Successfully!")

# -------------------------
# Accuracy Graph
# -------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training")

plt.plot(history.history["val_accuracy"], label="Validation")

plt.title("Accuracy")

plt.legend()

plt.savefig("outputs/accuracy.png")

plt.close()

# -------------------------
# Loss Graph
# -------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training")

plt.plot(history.history["val_loss"], label="Validation")

plt.title("Loss")

plt.legend()

plt.savefig("outputs/loss.png")

plt.close()

print("Graphs Saved!")