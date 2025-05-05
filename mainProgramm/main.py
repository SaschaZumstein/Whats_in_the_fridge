import cv2
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
ultralytics.checks()
from collections import Counter
import numpy as np
import threading
import queue
import os

confidence_threshold = 0.3
image_path = "./mainProgramm/output"
yolov8_path = "yolov8n.pt"
wanted_classes = [0, 39, 51]
yolo_trained_path = "./training_pepper_yoghurt/models/model_V3/weights/best.pt"
frame_skip = 5 
SHOW_WINDOWS = True

# ESP32-CAM Stream-URL
URL = "http://192.168.21.60"

# Load YOLO
model_yolo = YOLO(yolov8_path)
model_trained = YOLO(yolo_trained_path)

# Frame Queue
frame_queue = queue.Queue(maxsize=10)

# Start video-stream 
cap = cv2.VideoCapture(URL + ":81/stream")
if not cap.isOpened():
    print("Error: Cannot open the stream!")
    exit()

def is_bright_enough(image_gray, threshold=50):
    brightness = np.mean(image_gray)
    return brightness >= threshold

def extractClasses(results):
    return [results[0].names[int(box.cls)] for box in results[0].boxes]

def person_in_frame(classes):
    return ("person" in classes)

def is_not_blurry(image_gray, threshold=30.0):
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    return laplacian_var > threshold

def writeClassesToList(classes0, classes1):
    class_counts = Counter(classes0+classes1)
    with open(os.path.join(image_path, "list.txt"), "w", encoding="utf-8") as f:
        for cls, count in class_counts.items():
            f.write(f"{count}x: {cls}" + "\n")

def resize_for_display(frame):
    return cv2.resize(frame, (920, 690))

def get_color(cls_id):
    return tuple(int(c) for c in colors(cls_id))

def draw_combined_boxes(frame, results_list):
    overlay = frame.copy()
    for results in results_list:
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{results[0].names[cls_id]} {conf:.2f}"
            color = get_color(cls_id)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return overlay

# read the frames from the camera
def frame_reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No frame received!")
            break
        if not frame_queue.full():
            frame_queue.put(frame)

reader_thread = threading.Thread(target=frame_reader, daemon=True)
reader_thread.start()

# Main Loop
frame_cntr = 0
while True:
    if not frame_queue.empty():
        frame = frame_queue.get()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_bright_enough(frame_gray) and is_not_blurry(frame_gray): 
            # Only process every 5th frame
            frame_cntr += 1
            if frame_cntr % frame_skip != 0:
                continue

            # Predict with yolo
            results_yolo = model_yolo(frame, conf=confidence_threshold, classes=wanted_classes)
            classes_yolo = extractClasses(results_yolo)
            frame_yolo = results_yolo[0].plot()

            # check if no person is in image
            if not person_in_frame(classes_yolo):
                # predict with self trained modell
                results_trained = model_trained(frame, conf=confidence_threshold)
                classes_trained = extractClasses(results_trained)
                frame_trained = results_trained[0].plot()

                combined_frame = draw_combined_boxes(frame, [results_yolo, results_trained])

                cv2.imwrite(image_path + "/img.png", combined_frame)
                writeClassesToList(classes_yolo, classes_trained)
            else:
                combined_frame = frame_yolo

            # Debug only
            if SHOW_WINDOWS:
                cv2.imshow("Detection", resize_for_display(combined_frame))
        elif SHOW_WINDOWS:
            scaled_frame = resize_for_display(frame)
            cv2.putText(scaled_frame, "Image too dark", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(scaled_frame, "or too blurry", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Detection", scaled_frame)

    # escape with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()