import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd

# Load model
model = load_model('cnn_cifar10_model.h5')

# Label kelas CIFAR-10
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

# 🎨 App title & subtitle
st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🧠")
st.title("🧠 CIFAR-10 Image Classifier")
st.markdown("Upload an image and this app will predict which CIFAR-10 class it belongs to!")

# 📁 Upload file
uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    # Show original image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((32, 32))
    
    st.markdown("### 🖼️ Preview of Uploaded Image")
    st.image(image, use_container_width=True)

    # Preprocessing
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]  # shape: (10,)
    predicted_class = labels[np.argmax(prediction)]
    percent_values = [round(p * 100, 2) for p in prediction]

    # 🎯 Display top prediction
    st.success(f"### ✅ Predicted Class: **{predicted_class.upper()}** with {max(percent_values)}% confidence")

    # 🏆 Show top-3 predictions
    st.markdown("#### 🏅 Top 3 Predictions")
    top3_idx = np.argsort(prediction)[::-1][:3]
    for i in top3_idx:
        st.write(f"- **{labels[i].title()}**: {round(prediction[i]*100, 2)}%")

    # 📊 Show confidence as progress bars
    with st.expander("🔍 See All Class Probabilities"):
        for label, prob in zip(labels, percent_values):
            st.write(f"**{label.title()}**")
            st.progress(min(int(prob), 100))

    # 🧮 Option: Show bar chart
    with st.expander("📈 Show Prediction Bar Chart"):
        df = pd.DataFrame({
            'Class': labels,
            'Confidence (%)': percent_values
        })
        st.bar_chart(df.set_index("Class"))
