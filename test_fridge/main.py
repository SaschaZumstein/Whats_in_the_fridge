import cv2
import requests
import ultralytics
from ultralytics import YOLO
ultralytics.checks()
from collections import Counter
import numpy as np
import threading
import queue

confidence_threshold = 0.3
image_path = "./test_fridge/images"
yolov8_path = "yolov8n.pt"
wanted_classes = [0, 39, 51]
yolo_trained_path = "./training_pepper/models/model_V/weights/best.pt"

# ESP32-CAM Stream-URL
URL = "http://192.168.21.60"

# Load YOLO
model_yolo = YOLO(yolov8_path)
model_trained = YOLO(yolo_trained_path)

# Start video-stream 
cap = cv2.VideoCapture(URL + ":81/stream")
if not cap.isOpened():
    print("Error: Cannot open the stream!")
    exit()

def is_bright_enough(image_gray, threshold=50):
    brightness = np.mean(image_gray)
    return brightness >= threshold

def person_in_frame(results):
    detected_classes = [results[0].names[int(box.cls)] for box in results[0].boxes] 
    return ("person" in detected_classes)

def is_not_blurry(image_gray, threshold=100.0):
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    return laplacian_var > threshold

def writeClassesToList(results_yolo, results_trained):
    all_detected_classes = []

    detected_classes_yolo = [results_yolo[0].names[int(box.cls)] for box in results_yolo[0].boxes] 
    all_detected_classes.extend(detected_classes_yolo)

    detected_classes_trained = [results_trained[0].names[int(box.cls)] for box in results_trained[0].boxes]
    all_detected_classes.extend(detected_classes_trained)

    class_counts = Counter(all_detected_classes)
    with open(image_path + f"/list.txt", "w", encoding="utf-8") as f:
        for cls, count in class_counts.items():
            f.write(f"{count}x: {cls}" + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No frame received!")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if is_bright_enough(frame_gray) and is_not_blurry(frame_gray):
        # Predict with yolo
        results_yolo = model_yolo(frame, conf=confidence_threshold, classes=wanted_classes)
        annotated_frame_yolo = results_yolo[0].plot()

        # check if no person is in image
        if (not person_in_frame(results_yolo)):
            # predict with self trained modell
            results_trained = model_trained(frame, conf=confidence_threshold)
            annotated_frame_trained = results_trained[0].plot()

            cv2.imwrite(image_path + "/img_yolo.png", annotated_frame_yolo)
            cv2.imwrite(image_path + "/img_trained.png", annotated_frame_trained)
            writeClassesToList(results_yolo, results_trained)

        # For debug reason only
            cv2.imshow("YOLO Trained", annotated_frame_trained)
        cv2.imshow("YOLOv8", annotated_frame_yolo)
    else:
        cv2.putText(frame, "Image too dark", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "or too blurry", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("YOLOv8", frame)

    # escape with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()