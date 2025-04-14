import ultralytics
from ultralytics import YOLO
ultralytics.checks()
from collections import defaultdict

model_path = "./training_pepper/models/model_V"
image_dir = "./Test_Pictures"
confidence_threshold = 0.3

model = YOLO(model_path + "/weights/best.pt")
# Predict on an image
results = model(image_dir, save=True, project=model_path) 