import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# === Veri yolu ve yükleme aynı ===
data_dir = r"C:\Users\EXCALIBUR\Desktop\lidc_lungs_outputl5"
img_size = (128, 128)

images = []
masks = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.startswith("crop_") and file.endswith(".png"):
            img_path = os.path.join(root, file)
            mask_path = os.path.join(root, "mask_" + file.replace("crop_", ""))
            if os.path.exists(mask_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, img_size)
                mask = cv2.resize(mask, img_size)

                img = img / 255.0
                mask = (mask > 127).astype(np.float32)

                images.append(img)
                masks.append(mask)

images = np.expand_dims(np.array(images), axis=-1)
masks = np.expand_dims(np.array(masks), axis=-1)

print("Toplam görüntü sayısı:", images.shape[0])

X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)


# === U-Net modeli ===
def build_unet(input_shape=(128,128,1)):
    inputs = layers.Input(input_shape)

    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)

    u3 = layers.UpSampling2D()(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

    u1 = layers.UpSampling2D()(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(c6)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c6)

    return models.Model(inputs, outputs)


# === Hyperparameter Tuning ===
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]

results = {}

for lr in learning_rates:
    print(f"\n===== LR TEST EDİLİYOR: {lr} =====")

    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=8,
        epochs=5,
        verbose=1
    )

    val_loss = history.history["val_loss"][-1]
    val_acc = history.history["val_accuracy"][-1]

    results[lr] = (val_loss, val_acc)
    print(f"LR={lr} → val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


# === En iyi LR'yi bul ===
best_lr = min(results, key=lambda k: results[k][0])
print("\n============================================")
print(f"EN İYİ LEARNING RATE: {best_lr}")
print(f"Loss: {results[best_lr][0]:.4f}, Acc: {results[best_lr][1]:.4f}")
print("============================================")
