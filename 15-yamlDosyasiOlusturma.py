from ultralytics import YOLO

# YAML dosyası, YOLO gibi derin öğrenme modellerine veri setinin nerede olduğunu 
# ve nasıl yapılandığını tarif eden konfigürasyon dosyasıdır.

# yaml dosyasını oluşturmak:
yaml_content = """
path: C:/Users/EXCALIBUR/Desktop/lidc_yolo_dataset

train: images/train
val: images/val

names:
  0: nodule
"""

# yaml dosyasının adresi:
yaml_path = r"C:\Users\EXCALIBUR\Desktop\lidc_yolo_dataset\lidc_yolo.yaml"

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("YAML oluşturuldu:", yaml_path)

# YOLO modelini yükleme:
model = YOLO("yolov8n.pt")   # nano model → çok hızlı eğitir

# YOLO modelinin eğitimini düzenlemek ve başlatmak:
model.train(
    data=r"C:\Users\EXCALIBUR\Desktop\lidc_yolo_dataset\lidc_yolo.yaml",  # YAML dosyasının konumu
    epochs=10,        # Hızlı eğitim
    imgsz=256,        # Küçük çözünürlük → hızlı eğitim
    batch=8,          # CPU için ideal
    lr0=0.001,        # Öğrenme oranı
    device="cpu",     # GPU varsa "0" yazabilirsin
    name="lidc_yolo_fast"
)

print("\n📌 Eğitim tamamlandı!")
print("📦 Model kayıt konumu:")
print(r"C:\Users\EXCALIBUR\runs\detect\lidc_yolo_fast\weights\best.pt")
