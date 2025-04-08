import ultralytics
from ultralytics import YOLO
ultralytics.checks()
import cv2

model = YOLO("yolov8n.pt")

# Predict on an image
#results = model("C:/Users/FHGR/Documents/Bildverarbeitung_1/Software/object_detection/Pictures/gurke.jpg")
results = model("Software/object_detection/Pictures/karotte.jpeg")

# Plot the results
img = results[0].plot()
cv2.imshow("img", img)  # Display results
cv2.waitKey(0)