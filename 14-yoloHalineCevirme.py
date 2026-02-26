import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Ayarlar:
lidc_root = r"C:\Users\EXCALIBUR\.cache\kagglehub\datasets\zhangweiled\lidcidi\versions\1\LIDC-IDRI-slices"   # LIDC-IDRI-slices klasörün
yolo_root = r"C:\Users\EXCALIBUR\Desktop\lidc_yolo_dataset"  # YOLO
img_exts = (".png", ".jpg", ".jpeg", ".bmp")
test_size = 0.2  # train/val oranı

os.makedirs(os.path.join(yolo_root, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(yolo_root, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(yolo_root, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(yolo_root, "labels", "val"), exist_ok=True)

samples = []  # (img_path, mask_paths)

print("🔎 LIDC klasörü taranıyor...")

for patient in sorted(os.listdir(lidc_root)):
    p_dir = os.path.join(lidc_root, patient)
    if not os.path.isdir(p_dir):
        continue

    for nodule in sorted(os.listdir(p_dir)):
        n_dir = os.path.join(p_dir, nodule)
        images_dir = os.path.join(n_dir, "images")
        mask_dirs = [os.path.join(n_dir, f"mask-{i}") for i in range(4)]

        if not os.path.exists(images_dir):
            continue

        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(img_exts):
                continue

            img_path = os.path.join(images_dir, fname)
            mask_paths = [os.path.join(mdir, fname) for mdir in mask_dirs]

            samples.append((img_path, mask_paths))

print("Toplam slice sayısı:", len(samples))

# Train / Val ayır
train_samples, val_samples = train_test_split(samples, test_size=test_size, random_state=42)

def process_split(split_samples, split_name):
    # Çıkış görüntü ve etiket klasörlerinin yollarını belirle
    img_out_dir = os.path.join(yolo_root, "images", split_name)
    lbl_out_dir = os.path.join(yolo_root, "labels", split_name)

    idx = 0 # Kaydedilecek dosyalar için sayaç
    for img_path, mask_paths in split_samples:
        # Görüntüyü gri tonlamalı olarak oku
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape[:2] # Görüntünün yükseklil ve genişlik özellikleri

        # Maskeleri birleştir
        combined = np.zeros_like(img, np.uint8)
        mask_found = False

        for mp in mask_paths:
            if os.path.exists(mp): # Maske dosyası var mı?
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    # Maskeyi siyah-beyaz (binary) hale getir
                    _, m_bin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY) 
                    # Önceki maskelerle OR işlemi yaparak hepsini birleştir
                    combined = cv2.bitwise_or(combined, m_bin)
                    mask_found = True

        # Yeni dosya adı
        base_name = f"{split_name}_{idx:06d}"
        img_out_path = os.path.join(img_out_dir, base_name + ".png")
        lbl_out_path = os.path.join(lbl_out_dir, base_name + ".txt")

        # Grayscale'i 3 kanala çevirebilirsin (opsiyonel)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(img_out_path, img_bgr)

        # Label yaz
        if (not mask_found) or np.sum(combined) == 0:
            # Nodül yok -> boş label dosyası (negatif örnek)
            open(lbl_out_path, "w").close()
        else:
            # Maskedeki tüm beyaz piksellerin koordinatlarını al
            ys, xs = np.where(combined > 0)
            # bounding box sınırlarını belirle
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # YOLO formatına normalize et
            x_center = (x_min + x_max) / 2.0 / w
            y_center = (y_min + y_max) / 2.0 / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h

            # Sınıf id: 0 (nodule)
            with open(lbl_out_path, "w") as f:
                f.write(f"0 {x_center} {y_center} {bw} {bh}\n")

        idx += 1

    print(f"{split_name} için işlenen görüntü sayısı:", idx)


print("📦 Train set işleniyor...")
process_split(train_samples, "train")

print("📦 Val set işleniyor...")
process_split(val_samples, "val")

print("✅ YOLO dataset hazır!")
print("Konum:", yolo_root)
