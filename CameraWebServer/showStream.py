import cv2
import requests

# ESP32-CAM Stream-URL
URL = "http://192.168.204.60"

# Set resolution to UXGA (1600x1200)
requests.get(URL + "/control?var=framesize&val=10")

# Start video-stream
cap = cv2.VideoCapture(URL + ":81/stream")
if not cap.isOpened():
    print("Error: Cannot open the stream!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No frame received!")
        break
    
    cv2.imshow("ESP32-CAM Stream", frame)
    
    # escape with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()