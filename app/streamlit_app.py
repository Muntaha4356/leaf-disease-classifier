import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = 128

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

HEALTHY_ERROR_BASELINE = 0.00076

st.set_page_config(page_title="Plant Leaf Disease Classifier", page_icon="🌿")


@st.cache_resource
def load_all_models():
    scratch_cnn = load_model(os.path.join(APP_DIR, 'leaf_disease_cnn_v2.keras'))
    transfer_model = load_model(
        os.path.join(APP_DIR, 'leaf_disease_transfer_finetuned_balanced.keras'),
        custom_objects={'preprocess_input': preprocess_input}
    )
    autoencoder = load_model(os.path.join(APP_DIR, 'leaf_autoencoder.keras'))
    return scratch_cnn, transfer_model, autoencoder


def prepare_image(image: Image.Image):
    image = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype('float32')
    return np.expand_dims(arr, axis=0)


st.title("🌿 Plant Leaf Disease Classifier")
st.write(
    "Upload a photo of a plant leaf and pick a model to see its prediction. "
    "Three models are available, reflecting three different phases of this project — "
    "a from-scratch CNN, a fine-tuned transfer-learning model, and an unsupervised autoencoder."
)

scratch_cnn, transfer_model, autoencoder = load_all_models()

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
model_choice = st.radio(
    "Choose a model",
    [
        "Scratch CNN (Phase 1)",
        "Fine-tuned MobileNetV2 (Phase 2 — Best Model)",
        "Autoencoder Anomaly Detector (Phase 3 — Experimental)"
    ],
    index=1
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=300)

    if st.button("Analyze", type="primary"):
        batch = prepare_image(image)

        if model_choice == "Scratch CNN (Phase 1)":
            preds = scratch_cnn.predict(batch, verbose=0)[0]
            top_idx = np.argsort(preds)[::-1][:5]
            st.success(f"**Top prediction:** {CLASS_NAMES[top_idx[0]]} ({preds[top_idx[0]]*100:.1f}% confidence)")
            st.caption("Model: from-scratch CNN, trained without any pretrained weights.")
            for i in top_idx:
                st.write(f"{CLASS_NAMES[i]}")
                st.progress(float(preds[i]))

        elif model_choice == "Fine-tuned MobileNetV2 (Phase 2 — Best Model)":
            preds = transfer_model.predict(batch, verbose=0)[0]
            top_idx = np.argsort(preds)[::-1][:5]
            st.success(f"**Top prediction:** {CLASS_NAMES[top_idx[0]]} ({preds[top_idx[0]]*100:.1f}% confidence)")
            st.caption("Model: MobileNetV2, fine-tuned and class-balanced — 99% validation accuracy, the best-performing model in this project.")
            for i in top_idx:
                st.write(f"{CLASS_NAMES[i]}")
                st.progress(float(preds[i]))

        else:
            normalized = batch / 255.0
            reconstructed = autoencoder.predict(normalized, verbose=0)
            mse = float(np.mean(np.square(normalized - reconstructed)))
            ratio = mse / HEALTHY_ERROR_BASELINE
            verdict = "✅ Looks consistent with a healthy leaf" if ratio < 1.3 else "⚠️ Reconstruction error is elevated — possibly diseased"

            st.write(f"**Reconstruction error:** {mse:.5f} (healthy baseline ≈ {HEALTHY_ERROR_BASELINE:.5f})")
            st.info(verdict)
            st.warning(
                "This is an unsupervised anomaly detector trained only on healthy leaves, "
                "kept here to demonstrate the technique. In testing, it did not reliably separate "
                "healthy from diseased leaves as well as the supervised classifiers above — "
                "treat its output as experimental."
            )
