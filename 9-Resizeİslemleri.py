import os
import cv2
import numpy as np

# === Girdi ve çıktı yolları ===
root_path = r"C:\Users\EXCALIBUR\.cache\kagglehub\datasets\zhangweiled\lidcidri\versions\1\LIDC-IDRI-slices"
output_root = r"C:\Users\EXCALIBUR\Desktop\lidc_processed_256"
img_size = (256, 256)

os.makedirs(output_root, exist_ok=True)
out_img_dir = os.path.join(output_root, "images")
out_mask_dir = os.path.join(output_root, "masks")
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

idx = 0
num_skipped = 0

print("Hazır dataset oluşturuluyor...")

for patient in sorted(os.listdir(root_path)):
    patient_path = os.path.join(root_path, patient)
    if not os.path.isdir(patient_path):
        continue

    for nodule in sorted(os.listdir(patient_path)):
        nodule_path = os.path.join(patient_path, nodule)
        images_dir = os.path.join(nodule_path, "images")
        mask_dirs = [os.path.join(nodule_path, f"mask-{i}") for i in range(4)]

        if not os.path.exists(images_dir):
            continue

        for img_name in sorted(os.listdir(images_dir)):
            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # --- Maskeleri birleştir ---
            combined = np.zeros(img.shape, dtype=np.uint8)
            any_mask = False

            for mdir in mask_dirs:
                if not os.path.exists(mdir):
                    continue
                m_path = os.path.join(mdir, img_name)
                if not os.path.exists(m_path):
                    continue

                m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue

                # Maske boyutunu slice'a uydur
                if m.shape != img.shape:
                    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                _, m_bin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
                combined = cv2.bitwise_or(combined, m_bin)
                any_mask = True

            # Maske yoksa veya tamamen siyahsa atla
            if (not any_mask) or np.sum(combined) == 0:
                num_skipped += 1
                continue

            # ---- 256x256 resize + kaydetmeye hazırlık ----
            img_resized = cv2.resize(img, img_size)
            mask_resized = cv2.resize(combined, img_size, interpolation=cv2.INTER_NEAREST)
            mask_resized_bin = (mask_resized > 127).astype(np.uint8) * 255

            # Çıktı isimleri
            idx += 1
            img_out_path = os.path.join(out_img_dir, f"img_{idx:06d}.png")
            mask_out_path = os.path.join(out_mask_dir, f"mask_{idx:06d}.png")

            cv2.imwrite(img_out_path, img_resized)
            cv2.imwrite(mask_out_path, mask_resized_bin)

print("Toplam kaydedilen örnek:", idx)
print("Atlanan (maskesiz / boş maskeli) örnek:", num_skipped)
print("Hazır dataset klasörü:", output_root)
