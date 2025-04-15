import cv2
import requests
import ultralytics
from ultralytics import YOLO
ultralytics.checks()
from collections import Counter
import numpy as np

confidence_threshold = 0.3
index = 0
file_path = "./test_fridge/images"

# ESP32-CAM Stream-URL
URL = "http://192.168.204.60"

# Set resolution to UXGA (1600x1200)
requests.get(URL + "/control?var=framesize&val=10")

# Load YOLO
model = YOLO("yolov8n.pt")

# Start video-stream
cap = cv2.VideoCapture(URL + ":81/stream")
#cap = cv2.VideoCapture(0) # Test with laptop cam
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

def writeClassesToList(results, index):
    detected_classes = [results[0].names[int(box.cls)] for box in results[0].boxes] 
    class_counts = Counter(detected_classes)
    with open(file_path + f"/img_{index}.txt", "w", encoding="utf-8") as f:
        for cls, count in class_counts.items():
            f.write(f"{count}x: {cls}" + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No frame received!")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if is_bright_enough(frame_gray):
        # Predict on an image
        results = model(frame, conf=confidence_threshold)
        annotated_frame = results[0].plot()

        # check if person is in image
        if (not person_in_frame(results)) and is_not_blurry(frame_gray):
            cv2.imwrite(file_path + f"/img_{index}.png", annotated_frame)
            writeClassesToList(results, index)
            index += 1

        cv2.imshow("YOLOv8", annotated_frame)

    else:
        cv2.destroyAllWindows()

    # escape with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()