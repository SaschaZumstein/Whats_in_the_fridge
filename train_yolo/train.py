from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

results = model.train(
    data="C:/Bildverarbeitung_1/train_yolo/config.yaml",
    epochs=100,
    imgsz=640,
    batch=16,       # anpassen je nach RAM/GPU
    patience=20,    # early stopping falls nichts besser wird
    cache=True      # schnelleres Laden der Bilder
)