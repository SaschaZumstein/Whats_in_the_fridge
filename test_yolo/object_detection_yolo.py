import ultralytics
from ultralytics import YOLO
ultralytics.checks()
import cv2
import os

model_path = "./runs/detect/train2/weights/last.pt"
model = YOLO("yolov8n.pt")

image_dir = "./Pictures"

output_project = "./test_yolo/pictures_detected"

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)       

    # Predict on an image
    results = model(image_path, save=True, save_txt=False, save_conf=True, project=output_project, exist_ok=True)

print(f"Fertig! Ergebnisse in: {output_project}")