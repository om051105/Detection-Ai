import cv2
from ultralytics import YOLO
import time
import random

def turn_on_camera():
    model = YOLO("yolov8s.pt")  # Use a smaller model for better speed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    CONF_THRESHOLD = 0.5  # Confidence threshold
    NMS_THRESHOLD = 0.4   # Non-Maximum Suppression threshold

    color_dict = {}  # Dictionary to store unique colors for each object class

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        results = model(frame, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD)  # Perform object detection
        result = results[0]

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = result.names[cls]

            if conf < CONF_THRESHOLD:
                continue

            # Assign a unique color to each object class
            if label not in color_dict:
                color_dict[label] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

            color = color_dict[label]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    turn_on_camera()
