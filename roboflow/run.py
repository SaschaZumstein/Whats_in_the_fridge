from ultralytics import YOLO
import os
from pathlib import Path

# 🔹 1. Load the trained model
model_path = "runs/detect/train10/weights/best.pt"  # change if needed
model = YOLO(model_path)

# 🔹 2. Path to your test images
test_folder = "C:/Users/FHGR/Documents/Bildverarbeitung_1/roboflow/dataset_yoghurt/test"  # update this path

# 🔹 3. Run inference on the folder
results = model(test_folder, save=True)

# 🔹 4. Output folder (where YOLOv8 saves predictions)
output_dir = Path("runs/detect/predict")

# 🔹 5. Loop through results and print detections
for i, r in enumerate(results):
    print(f"\n🖼️ Image {i+1}: {r.path}")
    boxes = r.boxes.xyxy.cpu().numpy()   # (x1, y1, x2, y2)
    confs = r.boxes.conf.cpu().numpy()   # confidence scores
    classes = r.boxes.cls.cpu().numpy()  # class indices

    for box, conf, cls in zip(boxes, confs, classes):
        print(f"  ➤ Class: {model.names[int(cls)]}, Confidence: {conf:.2f}, Box: {box}")

print(f"\n✅ Predictions saved to: {output_dir.resolve()}")
