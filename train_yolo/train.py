from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

results = model.train(data="C:/Bildverarbeitung_1/train_yolo/config.yaml", epochs=100)