import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# ======================
#   Streamlit Arayüzü
# ======================

st.title("🫁 U-Net Lung Segmentation Demo")
st.write("Bu arayüz, eğittiğiniz U-Net modelini kullanarak akciğer segmentasyonu yapar.")

# ----------------------
# Model Yükleme
# ----------------------
model_path = r"C:\Users\EXCALIBUR\Desktop\unet_lung_segmentationml.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()
st.success("Model başarıyla yüklendi.")

# ----------------------
# Tahmin Fonksiyonu
# ----------------------
def preprocess_image(uploaded_file):
    """Görüntüyü yükle, 128x128'e indir, normalize et."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    img_resized = cv2.resize(img, (128, 128)) / 255.0
    img_input = np.expand_dims(np.expand_dims(img_resized, axis=-1), axis=0)

    return img, img_input  # orijinal ve işlenmiş döner

def predict_mask(model, img_input):
    pred = model.predict(img_input)[0, :, :, 0]
    pred_mask = (pred > 0.5).astype(np.uint8)
    return pred, pred_mask

# ----------------------
# Kullanıcıdan Görüntü Alma
# ----------------------
uploaded_file = st.file_uploader("Bir .png akciğer görüntüsü yükleyin", type=["png"])

if uploaded_file:
    st.subheader("Yüklenen Görüntü:")
    st.image(uploaded_file, width=300)

    original, img_input = preprocess_image(uploaded_file)

    # Tahmin
    pred, mask = predict_mask(model, img_input)

    # ----------------------
    # Sonuçları Gösterme
    # ----------------------
    st.subheader("Segmentation Sonucu")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Orijinal")
        st.image(original, clamp=True, width=250)

    with col2:
        st.write("Tahmin (Float Mask)")
        st.image(pred, clamp=True, width=250)

    with col3:
        st.write("Binarize Mask (0-1)")
        st.image(mask * 255, clamp=True, width=250)

# ----------------------
# Eğitim Grafiklerine Yer
# (İstersen history objesini kaydedersen buraya ekleyebilirim)
# ----------------------

st.info("Eğer eğitim history dosyasını (.npy) kaydedersen, eğitim grafiğini burada gösterebilirim.")
