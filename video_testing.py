from ultralytics import YOLO
import cv2

model = YOLO("best_new.pt")

cap = cv2.VideoCapture("bus4.mov")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()


results = model(frame)[0]

count = len(results.boxes)

print("People:", count)