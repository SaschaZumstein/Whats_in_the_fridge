import cv2
import requests

# ESP32-CAM Stream-URL
URL = "http://192.168.204.60"

# Auflösung auf UXGA (1600x1200) setzen
requests.get(URL + "/control?var=framesize&val=10")

# Video-Stream starten
cap = cv2.VideoCapture(URL + ":81/stream")
if not cap.isOpened():
    print("Fehler: Kann den Stream nicht öffnen!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler: Kein Frame empfangen!")
        break
    
    # In Graustufen umwandeln
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("ESP32-CAM Stream (Grayscale)", gray_frame)
    
    # Beenden mit q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()