from ultralytics import YOLO
import cv2

model = YOLO("best_new.pt")

# For webcam: cv2.VideoCapture(0)
# For video file: cv2.VideoCapture("bus_video.mp4")
# For IP camera: cv2.VideoCapture("rtsp://your_camera_ip")
# cap = cv2.VideoCapture(0)
cap=cv2.VideoCapture("bus3.mp3")

BUS_CAPACITY = 60  # set your bus max capacity

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)
    count = len(results[0].boxes)

    # Density calculation
    occupancy = count / BUS_CAPACITY
    if occupancy < 0.5:
        density = "LOW"
        color = (0, 255, 0)      # green
    elif occupancy < 0.8:
        density = "MEDIUM"
        color = (0, 165, 255)    # orange
    else:
        density = "HIGH"
        color = (0, 0, 255)      # red

    # Draw detections
    annotated = results[0].plot()

    # Overlay info
    cv2.rectangle(annotated, (0, 0), (400, 80), (0, 0, 0), -1)
    cv2.putText(annotated, f"People: {count}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(annotated, f"Density: {density} ({occupancy:.0%})", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Bus Occupancy Monitor", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()