import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# veri yolunu belirleme:
data_dir = r"C:\Users\EXCALIBUR\Desktop\lidc_lungs_output5"
img_size = (128, 128)

images, masks = [], []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.startswith("crop_") and file.endswith(".png"):
            img_path = os.path.join(root, file)
            mask_path = os.path.join(root, "mask_" + file.replace("crop_", ""))
            if os.path.exists(mask_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size) / 255.0
                mask = (cv2.resize(mask, img_size) > 127).astype(np.float32)
                images.append(img)
                masks.append(mask)

images = np.expand_dims(np.array(images), axis=-1)
masks = np.expand_dims(np.array(masks), axis=-1)

print("Toplam görüntü sayısı:", images.shape[0])

# Veri Setlerini ayırmak:

X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# U-Net Modelin mimarisi:

def build_unet(filters=32, lr=0.001, dropout_rate=0.6):
    inputs = layers.Input((128,128,1))

    # Encoder
    c1 = layers.Conv2D(filters, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(filters, 3, activation='relu', padding='same')(c1)
    c1 = layers.Dropout(dropout_rate)(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(filters*2, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(filters*2, 3, activation='relu', padding='same')(c2)
    c2 = layers.Dropout(dropout_rate)(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(filters*4, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(filters*4, 3, activation='relu', padding='same')(c3)
    c3 = layers.Dropout(dropout_rate)(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    # Bottleneck
    bn = layers.Conv2D(filters*8, 3, activation='relu', padding='same')(p3)
    bn = layers.Conv2D(filters*8, 3, activation='relu', padding='same')(bn)
    bn = layers.Dropout(dropout_rate)(bn)

    # Decoder
    u3 = layers.UpSampling2D()(bn)
    u3 = layers.Concatenate()([u3, c3])
    c4 = layers.Conv2D(filters*4, 3, activation='relu', padding='same')(u3)
    c4 = layers.Conv2D(filters*4, 3, activation='relu', padding='same')(c4)
    c4 = layers.Dropout(dropout_rate)(c4)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(filters*2, 3, activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(filters*2, 3, activation='relu', padding='same')(c5)
    c5 = layers.Dropout(dropout_rate)(c5)

    u1 = layers.UpSampling2D()(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = layers.Conv2D(filters, 3, activation='relu', padding='same')(u1)
    c6 = layers.Conv2D(filters, 3, activation='relu', padding='same')(c6)
    c6 = layers.Dropout(dropout_rate)(c6)

    output = layers.Conv2D(1, 1, activation='sigmoid')(c6)

    model = models.Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Hipermetre grid aramsının yapıldığı yer:
learning_rate = 0.0005
batch_size = 8
filters = 16
dropout_rate = 0.3
epochs = 10

# Model Eğitimi:
model = build_unet(filters=filters, lr=learning_rate, dropout_rate=dropout_rate)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# Modelin kaydedilmesi:
model.save(r"C:\Users\EXCALIBUR\Desktop\unet_lung_segmentationml.h5")
print("\nModel kaydedildi.")

# Eğitim Sonuçlarının Görselleştirilmesi:
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Doğruluk')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Kayıp')
plt.show()
