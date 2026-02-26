import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# === AYARLAR ===
root_path = r"C:\Users\EXCALIBUR\.cache\kagglehub\datasets\zhangweiled\lidcidri\versions\1\LIDC-IDRI-slices"
output_root = r"C:\Users\EXCALIBUR\Desktop\lidc_lungs_outputl5"
resize_to = None  # Orijinal çözünürlük korunacak
image_exts = (".png", ".jpg", ".jpeg", ".bmp")
# ===============

os.makedirs(output_root, exist_ok=True)

cnt_images = 0
cnt_processed = 0
cnt_missing_mask = 0
cnt_roi_skipped = 0

for patient in sorted(os.listdir(root_path)):
    patient_path = os.path.join(root_path, patient)
    if not os.path.isdir(patient_path):
        continue

    out_patient_dir = os.path.join(output_root, patient)
    os.makedirs(out_patient_dir, exist_ok=True)

    for nodule in sorted(os.listdir(patient_path)):
        nodule_path = os.path.join(patient_path, nodule)
        images_dir = os.path.join(nodule_path, "images")
        mask_dirs = [os.path.join(nodule_path, f"mask-{i}") for i in range(4)]

        if not os.path.exists(images_dir):
            continue

        out_nodule_dir = os.path.join(out_patient_dir, nodule)
        os.makedirs(out_nodule_dir, exist_ok=True)

        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(image_exts)])
        if len(image_files) == 0:
            continue

        for img_file in image_files:
            cnt_images += 1
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Okunamadı (image):", img_path)
                continue

            combined_mask = np.zeros_like(img, dtype=np.uint8)
            any_mask_found = False

            for mdir in mask_dirs:
                if not os.path.exists(mdir):
                    continue
                mask_path_candidate = os.path.join(mdir, img_file)
                if os.path.exists(mask_path_candidate):
                    m = cv2.imread(mask_path_candidate, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    if m.shape != img.shape:
                        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    _, m_bin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
                    combined_mask = cv2.bitwise_or(combined_mask, m_bin)
                    any_mask_found = True

            # Maskesi tamamen siyahsa atla
            if not any_mask_found or np.sum(combined_mask) == 0:
                cnt_missing_mask += 1
                continue

            combined_mask_bin = (combined_mask > 127).astype(np.uint8)
            lung_region = (img.astype(np.float32) * combined_mask_bin).astype(np.uint8)

            # Gürültü Azaltma + Kontrast Artırma
            lung_denoised = cv2.bilateralFilter(lung_region, 9, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lung_enhanced = clahe.apply(lung_denoised)

            # ROI (mask bounding box) çıkarımı
            ys, xs = np.where(combined_mask_bin > 0)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = np.min(xs), np.max(xs)
                y_min, y_max = np.min(ys), np.max(ys)
                roi_crop = lung_enhanced[y_min:y_max, x_min:x_max]
                # ROI boşsa atla
                if roi_crop.size == 0:
                    cnt_roi_skipped += 1
                    continue
            else:
                cnt_roi_skipped += 1
                continue

            # Çıktı dosyaları
            out_img_path = os.path.join(out_nodule_dir, img_file)
            out_mask_path = os.path.join(out_nodule_dir, "mask_" + img_file)
            out_crop_path = os.path.join(out_nodule_dir, "crop_" + img_file)

            cv2.imwrite(out_img_path, lung_enhanced)
            cv2.imwrite(out_mask_path, (combined_mask_bin * 255).astype(np.uint8))
            cv2.imwrite(out_crop_path, roi_crop)

            cnt_processed += 1

print("Toplam görüntü sayısı:", cnt_images)
print("İşlenen ve kaydedilen:", cnt_processed)
print("Mask bulunmayan/siyah maskeler:", cnt_missing_mask)
print("Boş ROI atlananlar:", cnt_roi_skipped)
print("Çıktılar kaydedildi ->", output_root)
