# -*- coding: utf-8 -*-
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import zipfile
import os
import tempfile
import glob
import json
from tensorflow.keras.callbacks import Callback

# -----------------------------
# Settings
# -----------------------------
IMG_WIDTH, IMG_HEIGHT = 256, 256
BATCH_SIZE = 32

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Progress bar callback
# -----------------------------
class StreamlitProgress(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
    
    def on_epoch_end(self, epoch, logs=None):
        p = (epoch + 1) / self.epochs
        self.progress_bar.progress(p)
        self.status_text.text(
            f"Epoch {epoch+1}/{self.epochs} | "
            f"loss={logs['loss']:.4f} | "
            f"val_loss={logs.get('val_loss',0):.4f} | "
            f"acc={logs.get('accuracy',0):.4f} | "
            f"val_acc={logs.get('val_accuracy',0):.4f}"
        )

# -----------------------------
# Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])

# -----------------------------
# CNN model
# -----------------------------
def create_cnn_model(num_classes):
    return Sequential([
        layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

# -----------------------------
# Create dataset
# -----------------------------
def create_datasets(data_dir):
    class_names = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])
    num_classes = len(class_names)

    train = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    train = train.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val = val.cache().prefetch(tf.data.AUTOTUNE)

    return train, val, class_names, num_classes

# -----------------------------
# List saved models
# -----------------------------
def get_saved_models():
    return [os.path.basename(f) for f in glob.glob(os.path.join(MODEL_DIR, "*.h5"))]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Fruit Classifier 🍎🍌🍓")
tabs = st.tabs(["Train Model", "Inference"])

# ================================================================
# TRAIN MODEL
# ================================================================
with tabs[0]:
    st.header("Train a New Model")

    uploaded_zip = st.file_uploader("Upload dataset ZIP", type=["zip"])
    epochs = st.slider("Epochs", 1, 50, 5)
    batch_size = st.select_slider("Batch size", [16, 32, 64], value=32)

    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "dataset.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getvalue())
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)

            # Проверяем наличие классов
            class_dirs = [d for d in os.listdir(tmpdir)
                          if os.path.isdir(os.path.join(tmpdir, d))]
            if len(class_dirs) < 1:
                st.error("Dataset must contain at least 1 class folder.")
                st.stop()

            train_ds, val_ds, class_names, num_classes = create_datasets(tmpdir)
            st.write("Detected classes:", class_names)

            model = create_cnn_model(num_classes)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            model_name = st.text_input("Model name", "my_model")

            if st.button("Start Training", key="train_start"):
                progress = StreamlitProgress(epochs)
                model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[progress])

                # Сохраняем модель в .h5
                save_path = os.path.join(MODEL_DIR, model_name + ".h5")
                model.save(save_path, save_format="h5")

                # Сохраняем имена классов рядом с моделью
                class_file = os.path.join(MODEL_DIR, model_name + "_classes.json")
                with open(class_file, "w") as f:
                    json.dump(class_names, f)

                st.success(f"Model and class names saved:\n{save_path}\n{class_file}")

# ================================================================
# INFERENCE
# ================================================================
with tabs[1]:
    st.header("Model Prediction")

    models_list = get_saved_models()
    selected = st.selectbox("Choose model", models_list)

    if selected:
        full_path = os.path.join(MODEL_DIR, selected)
        model = tf.keras.models.load_model(full_path)
        st.success(f"Loaded model: {selected}")

        # Загружаем имена классов
        class_file = full_path.replace(".h5", "_classes.json")
        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                class_names = json.load(f)
        else:
            # fallback
            num_classes = model.output_shape[-1]
            class_names = [f"class_{i}" for i in range(num_classes)]

        img_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

        if img_file:
            img = Image.open(img_file).convert("RGB")
            st.image(img, use_column_width=True)

            resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
            arr = np.expand_dims(np.array(resized)/255.0, axis=0)

            pred = model.predict(arr)[0]
            idx = np.argmax(pred)

            st.success(f"Prediction: {class_names[idx]} ({pred[idx]:.2%})")

        if st.button("Delete model"):
            os.remove(full_path)
            class_file = full_path.replace(".h5", "_classes.json")
            if os.path.exists(class_file):
                os.remove(class_file)
            st.warning("Model and class names removed.")
 
