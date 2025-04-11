import os
import shutil

train_dir = "./openImage/dataset/labels/train"
val_dir = "./openImage/dataset/labels/val"

# Alle Bild-Dateien alphabetisch sortieren
images = sorted([f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))])

# Berechne Index für letzten 20%
split_index = int(len(images) * 0.8)
val_images = images[split_index:]

# Verschiebe Bilder
for img in val_images:
    src = os.path.join(train_dir, img)
    dst = os.path.join(val_dir, img)
    shutil.move(src, dst)

print(f"{len(val_images)} Bilder verschoben von '{train_dir}' nach '{val_dir}'.")