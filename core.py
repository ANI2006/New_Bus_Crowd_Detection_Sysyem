from ultralytics import YOLO
import cv2

model = YOLO("best_new.pt")


cap=cv2.VideoCapture("bus3.mp3")

BUS_CAPACITY = 60

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)
    count = len(results[0].boxes)

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

    annotated = results[0].plot()

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