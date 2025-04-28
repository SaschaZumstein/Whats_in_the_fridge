from ultralytics import YOLO

config_path = "./training_yoghurt/config.yaml"
model_path = "./training_yoghurt/models"

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

results = model.train(data=config_path, epochs=150, project=model_path, name="model_V") 