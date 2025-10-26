import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound
import threading

MODEL_PATH = "yolov8n.pt"  # Modelo YOLO
MOVEMENT_THRESHOLD = 25
BEEP_PATH = "beep.mp3"
OBJECT_TO_WATCH = "person"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("❌ Não foi possível acessar a câmera.")

def beep():
    threading.Thread(target=playsound, args=(BEEP_PATH,), daemon=True).start()

previous_positions = {}
print("✅ Sistema iniciado. Pressione 'Q' para sair.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if label in previous_positions:
                px, py = previous_positions[label]
                movement = np.sqrt((cx - px)**2 + (cy - py)**2)

                if movement > MOVEMENT_THRESHOLD:
                    print(f"⚠️ Movimento detectado em {label} ({movement:.1f})")
                    if label == OBJECT_TO_WATCH:
                        beep()

            previous_positions[label] = (cx, cy)

    cv2.imshow("Detector de Movimento - YOLOv8", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Sistema encerrado.")
