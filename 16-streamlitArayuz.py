import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import os

# Ayarlar:
yolo_path = r"C:\Users\EXCALIBUR\.spyder-py3\runs\detect\lidc_yolo_fast2\weights\best.pt"
save_crop_dir = r"C:\Users\EXCALIBUR\Desktop\detected_crops"

os.makedirs(save_crop_dir, exist_ok=True)

# Modelin arayüze yüklenmesi
@st.cache_resource
def load_yolo():
    return YOLO(yolo_path)

yolo_model = load_yolo()
st.success("YOLO modeli yüklendi.")

# YOLO tespitinin çalışması:
def detect_and_crop(img, save_dir):
    """YOLO ile nodülü bulur, crop eder ve kaydeder."""
    
    # YOLO 3 kanal ister → dönüştür
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    results = yolo_model(img_color, verbose=False)[0]

    if results.boxes.xyxy is None or len(results.boxes) == 0:
        return None, None

    boxes = results.boxes.xyxy.cpu().numpy()

    # En büyük kutuyu seç
    areas = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in boxes]
    best = boxes[np.argmax(areas)]
    x1, y1, x2, y2 = best.astype(int)

    # Crop
    crop = img[y1:y2, x1:x2]

    # Kaydet
    save_path = os.path.join(save_dir, f"crop_{np.random.randint(1e9)}.png")
    cv2.imwrite(save_path, crop)

    return (x1, y1, x2, y2), save_path


# Streamlit Arayüz:
st.title("🫁 YOLO Lung Nodule Detector")
uploaded = st.file_uploader("Bir CT slice (.png) yükleyin", type=["png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.subheader("Yüklenen Görüntü")
    st.image(img, width=350)

    bbox, saved_path = detect_and_crop(img, save_crop_dir)

    if bbox is None:
        st.error("❌ YOLO nodül tespit edemedi.")
    else:
        x1, y1, x2, y2 = bbox

        img_box = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_box, (x1,y1), (x2,y2), (0,255,0), 2)

        st.subheader("🟩 YOLO Tespit Sonucu")
        st.image(img_box, width=350)

        st.success(f"✔ Crop kaydedildi: {saved_path}")

