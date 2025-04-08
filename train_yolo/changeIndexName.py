import os

# Pfad zum Ordner mit den Label-Dateien
labels_dir = "./train_yolo/dataset_eggs/labels/train"

# Durchlaufe alle .txt-Dateien im Label-Ordner
for label_file in os.listdir(labels_dir):
    if label_file.endswith(".txt"):
        # Vollständigen Pfad zur Datei
        label_path = os.path.join(labels_dir, label_file)

        # Öffne die Label-Datei und ersetze "Egg" durch die ID "0"
        with open(label_path, "r") as file:
            lines = file.readlines()

        # Ersetze "Egg" mit "0"
        with open(label_path, "w") as file:
            for line in lines:
                file.write(line.replace("Egg", "0"))

print("Alle 'Egg'-Labels wurden zu '0' umgewandelt.")