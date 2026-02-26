import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


root_path = r"C:\Users\EXCALIBUR\.cache\kagglehub\datasets\zhangweiled\lidcidri\versions\1\LIDC-IDRI-slices"
output_root = r"C:\Users\EXCALIBUR\Desktop\lidc_lungs_output5"
resize_to = None  # Orijinal çözünürlük korunacak
image_exts = (".png", ".jpg", ".jpeg", ".bmp")

all_lungs = []
for root, dirs, files in os.walk(output_root):
    for f in files:
        if f.lower().endswith(image_exts) and not f.startswith(("mask_", "crop_")):
            all_lungs.append(os.path.join(root, f))

if len(all_lungs) == 0:
    print("Hiç çıktı görüntüsü bulunamadı!")
else:
    random_img_path = random.choice(all_lungs)
    random_mask_path = os.path.join(os.path.dirname(random_img_path), "mask_" + os.path.basename(random_img_path))
    random_crop_path = os.path.join(os.path.dirname(random_img_path), "crop_" + os.path.basename(random_img_path))

    img = cv2.imread(random_img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(random_mask_path, cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(random_crop_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.title("İşlenmiş Görüntü")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Crop (Nodül ROI)")
    plt.imshow(crop, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



def augment_image(img):
    # Temel veri artırma: yatay/ters çevirme, döndürme, parlaklık değişimi
    ops = [
        lambda x: cv2.flip(x, 1),  # Yatay çevirme
        lambda x: cv2.flip(x, 0),  # Dikey çevirme
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),# Çevirme
        lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),# Çevirme
        lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=20),  # Kontrast ve parlaklık artırma
    ]
    op = random.choice(ops)
    return op(img)

# --- Augmentation öncesi toplam crop sayısını say ---
crop_images = []
for root, dirs, files in os.walk(output_root):
    for f in files:
        if f.lower().startswith("crop_") and f.lower().endswith(image_exts):
            crop_images.append(os.path.join(root, f))

original_count = len(crop_images)
print(f"\nToplam {original_count} adet crop (ROI) görüntüsü bulundu. Augmentation başlıyor...")

# --- Augmentation işlemi ---
aug_count = 0
for crop_path in crop_images:
    img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    for i in range(2):  # Her crop'tan 2 yeni varyasyon oluştur
        aug_img = augment_image(img)
        new_name = crop_path.replace(".png", f"_aug{i+1}.png")
        cv2.imwrite(new_name, aug_img)
        aug_count += 1

# --- Augmentation sonrası sayım ---
total_after = original_count + aug_count
print(f"Augmentation tamamlandı!\nYeni oluşturulan görüntü sayısı: {aug_count}\nToplam görüntü sayısı: {total_after}")

