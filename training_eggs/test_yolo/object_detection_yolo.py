import ultralytics
from ultralytics import YOLO
ultralytics.checks()
import cv2
import os
import numpy as np


model_path = "./runs/detect/train2/weights/last.pt"
model_pretrained = YOLO("yolov8n.pt")
model_selftrained = YOLO(model_path)

image_dir = "./Pictures"

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)       

    # Predict on an image
    results_pretrained = model_pretrained(image_path)
    results_selftrained = model_selftrained(image_path)

    img_pretrained = results_pretrained[0].plot()
    img_selftrained = results_selftrained[0].plot()

    combined = np.hstack((img_pretrained, img_selftrained))

    cv2.imwrite(f"test_yolo/output/vergleich_{filename}.jpg", combined)