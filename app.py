import streamlit as st
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model.h5")

st.title("Handwritten Digit Recognition")

uploaded = st.file_uploader(
    "Upload a 28x28 grayscale digit image",
    type=["png","jpg","jpeg"]
)

if uploaded:
    from PIL import Image

    image = Image.open(uploaded).convert("L")
    image = image.resize((28,28))

    img=np.array(image)/255.0
    img=img.reshape(1,28,28)

    prediction=model.predict(img)
    digit=np.argmax(prediction)

    st.success(f"Predicted Digit: {digit}")
