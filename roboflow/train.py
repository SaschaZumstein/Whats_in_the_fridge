from ultralytics import YOLO

model = YOLO("yolov8n.pt")

result = model.train(data="C:/Users/FHGR/Documents/Bildverarbeitung_1/roboflow/dataset_yoghurt/data.yaml", epochs=100)