import streamlit as st
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('processed_cancer.csv')

df = load_data()

# Model selection
MODEL_PATHS = {
    "Sequencial model": "models/cancer_seq_88.pkl",
    "Efficientnet model": "models/cancer_efficientnet_87.pkl",
    # "Model 3": "models/cancer_mobilenet_64.pkl",
    # "Model 4": "models/cancer_resetnet_73.pkl",
    # "Model 5": "models/cancer_resetnet_87.pkl",
    # "Model 6": "models/cancer_vgg16_76.pkl",
    # "Model 7": "models/history_vgg16.pkl"
}


# Load the selected model
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Sample images
sample_images = {
    "Sample 1": "random_test_imgs/0.jpeg",
    "Sample 2": "random_test_imgs/3.jpg",
    "Sample 3": "random_test_imgs/1.3.jpeg",
    "Sample 4": "random_test_imgs/1.jpg"
}

st.title("🩺 Cancer Cell Type Classification")
st.markdown("---")

# Model selection with buttons
st.subheader("Choose a Model")
selected_model = st.radio("", list(MODEL_PATHS.keys()), horizontal=True)
model = load_model(MODEL_PATHS[selected_model])

image_size = (64, 64) if selected_model == "Sequencial model" else (224, 224)

# Display sample images
st.subheader("Choose a Sample Image")
selected_sample = st.selectbox("Select an image", list(sample_images.keys()))
sample_path = sample_images[selected_sample]

# Load and resise the selected image
img = Image.open(sample_path).resize(image_size)

# Convert image to array
x = np.asarray(img)
x = np.expand_dims(x, axis=0)

# Prediction
pred = model.predict(x)
prediction = np.argmax(pred, axis=1)[0]

label_map={
    0:'Actinic keratoses',
    1:'Basal cell carcinoma',
    2:'Benign keratosis-like lesions ',
    3:'Dermatofibroma',
    4:'Melanocytic nevi',
    5:'Melanoma',
    6:'Vascular lesions',
}

prediction=label_map[prediction]

# Display results
col1, col2 = st.columns([1, 2])
with col1:
    st.image(img, caption="🔬 Processed Image", width=100)
with col2:
    st.markdown(
        f"<h2 style='color: green;'>Predicted Class:<br>{prediction}</h2>",
        unsafe_allow_html=True
    )
