from ultralytics import YOLO

# start from begin

config_path = "./training_pepper_yoghurt/config.yaml"
model_path = "./training_pepper_yoghurt/models"

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

results = model.train(data=config_path, epochs=200, project=model_path, name="model_V")

# start with already trained data

model = YOLO("./training_pepper/models/model_V/weights/last.pt")
model.train(resume=True)