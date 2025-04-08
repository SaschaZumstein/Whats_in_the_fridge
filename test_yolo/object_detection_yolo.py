import ultralytics
from ultralytics import YOLO
ultralytics.checks()
import cv2

model_path = "C:/Bildverarbeitung_1/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Predict on an image
results = model("C:/Bildverarbeitung_1/test_yolo/karotte.jpeg")

# Plot the results
img = results[0].plot()
cv2.imshow("img", img)  # Display results
cv2.waitKey(0)