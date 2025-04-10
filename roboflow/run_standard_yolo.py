import ultralytics
from ultralytics import YOLO
ultralytics.checks()

import os
import cv2
import matplotlib.pyplot as plt

#model = YOLO("yolo11n.pt")

def plot_results(results):
    img = results[0].plot()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert image to RGB
    plt.imshow(img_rgb)  # Display results
    plt.axis('off')  # Hide axes
    plt.show()

model = YOLO("yolov8n.pt")

# Predict on an image
#results = model("C:/Users/FHGR/Documents/Bildverarbeitung_1/Software/object_detection/Pictures/gurke.jpg")
#results = model("Software/object_detection/Pictures/karotte.jpeg")
test_folder = "C:/Users/FHGR/Documents/Bildverarbeitung_1/roboflow/dataset_yoghurt/test"  # update this path

results = model(test_folder, save=True)


# Plot the results
plot_results(results)