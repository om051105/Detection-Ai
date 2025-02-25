import cv2
import torch
from ultralytics import YOLO
import threading

model = YOLO("yolov8n.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

def process_frame(frame):
    return model(frame, conf=0.5, iou=0.45, imgsz=640)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    thread = threading.Thread(target=process_frame, args=(frame,))
    thread.start()
    results = process_frame(frame)

    count = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(float(box.conf[0]), 2)
            label = result.names[int(box.cls[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            count += 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"Objects: {count} | FPS: {int(fps)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Fast Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
