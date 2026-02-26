import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ============================
# 1) Dataset yolları
# ============================

data_root = r"C:\Users\EXCALIBUR\Desktop\lidc_processed_256"
img_dir = os.path.join(data_root, "images")
mask_dir = os.path.join(data_root, "masks")

img_files = sorted(os.listdir(img_dir))

images = []
masks = []

print("Dataset yükleniyor...")

# ============================
# 2) Görüntü ve maskeleri yükle
# ============================

for f in img_files:
    img_path = os.path.join(img_dir, f)
    mask_path = os.path.join(mask_dir, "mask_" + f.split("img_")[-1])

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        continue

    img = img.astype(np.float32) / 255.0
    mask = (mask > 127).astype(np.float32)

    images.append(img[..., np.newaxis])
    masks.append(mask[..., np.newaxis])

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

print("Toplam örnek sayısı:", images.shape[0])

# ============================
# 3) Train/Validation ayırma
# ============================

X_train, X_val, y_train, y_val = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)

# ============================
# 4) Dice Loss + BCE
# ============================

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2 * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# ============================
# 5) U-NET Modeli
# ============================

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    return x

def build_unet():
    inputs = layers.Input((256, 256, 1))

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    bn = conv_block(p4, 512)

    u4 = layers.UpSampling2D()(bn)
    u4 = layers.concatenate([u4, c4])
    c5 = conv_block(u4, 256)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.concatenate([u3, c3])
    c6 = conv_block(u3, 128)

    u2 = layers.UpSampling2D()(c6)
    u2 = layers.concatenate([u2, c2])
    c7 = conv_block(u2, 64)

    u1 = layers.UpSampling2D()(c7)
    u1 = layers.concatenate([u1, c1])
    c8 = conv_block(u1, 32)

    output = layers.Conv2D(1, 1, activation="sigmoid")(c8)

    model = models.Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0003),
        loss=bce_dice_loss,
        metrics=["accuracy"]
    )
    return model

model = build_unet()
model.summary()

# ============================
# 6) Model Eğitimi
# ============================

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=8,
    validation_data=(X_val, y_val),
    verbose=1
)

model.save(r"C:\Users\EXCALIBUR\Desktop\unet_lidc256.h5")
print("Model kaydedildi!")

# ============================
# 7) Eğitim Grafikleri
# ============================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train", "Val"])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["Train", "Val"])

plt.show()
