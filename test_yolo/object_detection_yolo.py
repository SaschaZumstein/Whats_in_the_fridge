import ultralytics
from ultralytics import YOLO
ultralytics.checks()
import cv2
import os

model_path = "./runs/detect/train2/weights/last.pt"
model = YOLO("yolov8n.pt")

image_dir = "./Pictures"

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, filename)       

    # Predict on an image
    results = model(image_path)

    # Plot the results
    img = results[0].plot()
    cv2.imshow("img", img)  # Display results
    cv2.waitKey(0)