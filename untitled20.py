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
import time

# -----------------------------
# Settings
# -----------------------------
IMG_WIDTH, IMG_HEIGHT = 256, 256
BATCH_SIZE = 32
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Predefined class names
# -----------------------------
freshness_classes = ["Fresh", "Rotten"]
fruit_classes = ["Apple", "Banana", "Strawberry"]

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
def get_main_data_dir(tmpdir):
    # Все папки верхнего уровня
    top_folders = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
    if len(top_folders) == 0:
        st.error("No folders found in ZIP!")
        st.stop()

    # Проверяем __MACOSX
    if top_folders[0] == "__MACOSX":
        if len(top_folders) < 2:
            st.error("No valid class folders found in ZIP!")
            st.stop()
        main_dir = os.path.join(tmpdir, top_folders[1])
    else:
        main_dir = os.path.join(tmpdir, top_folders[0])

    return main_dir


def get_data_dir_for_dataset(tmpdir):
    # Список папок верхнего уровня
    top_folders = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
    if "__MACOSX" in top_folders:
        # Берём любую папку кроме __MACOSX
        top_folders = [d for d in top_folders if d != "__MACOSX"]
        if not top_folders:
            st.error("No valid folders found in ZIP!")
            st.stop()
        data_dir = os.path.join(tmpdir, top_folders[0])
    else:
        # Если __MACOSX нет, просто используем tmpdir
        data_dir = tmpdir

    return data_dir


def create_datasets(tmpdir):
    data_dir = get_data_dir_for_dataset(tmpdir)

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

    class_names = train.class_names
    num_classes = len(class_names)

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

            train_ds, val_ds, class_names, num_classes = create_datasets(tmpdir)
            st.write("Detected classes:", class_names)

            model = create_cnn_model(num_classes)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            model_name = st.text_input("Model name", "my_model")

            if st.button("Start Training", key="train_start"):
                progress = StreamlitProgress(epochs)
                model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[progress])

                # Сохраняем модель и JSON с классами
                save_path = os.path.join(MODEL_DIR, model_name + ".h5")
                model.save(save_path, save_format="h5")
                class_file = os.path.join(MODEL_DIR, model_name + "_classes.json")
                with open(class_file, "w") as f:
                    json.dump(class_names, f)

                st.success(f"Model saved: {save_path}")
                st.success(f"Class names saved: {class_file}")

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

        # Жестко заданные классы для предзагруженных моделей
        if "freshness_model" in selected.lower():
            class_names = freshness_classes
        elif "fruit_model" in selected.lower():
            class_names = fruit_classes
        else:
            # Попытка загрузить JSON для пользовательских моделей
            class_file = full_path.replace(".h5", "_classes.json")
            if os.path.exists(class_file):
                with open(class_file, "r") as f:
                    class_names = json.load(f)
            else:
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
        from streamlit_extras.stylable_container import stylable_container

        # --- Red delete button ---
        with stylable_container(
            key="delete_style",
            css_styles="""
                button {
                    background-color: #FF4D4D !important;
                    color: white !important;
                    border-radius: 8px !important;
                    width: 100% !important;
                    height: 45px !important;
                    font-weight: 600 !important;
                    border: none !important;
                }
                button:hover {
                    background-color: #E60000 !important;
                }
            """
        ):
            delete_clicked = st.button("Delete model", key="delete_model")
        if delete_clicked:
            status = st.warning("Deleting model…")   # Step 1: show message
            time.sleep(4)                            # Step 2: delay
            
            try:
                os.remove(full_path)
                class_file = full_path.replace(".h5", "_classes.json")
                if os.path.exists(class_file):
                    os.remove(class_file)
        
                status.success("Model deleted!")     # Step 3: update message
            except Exception as e:
                status.error(f"Error deleting model: {e}")
