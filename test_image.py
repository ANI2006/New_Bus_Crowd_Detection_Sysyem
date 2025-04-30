from ultralytics import YOLO

model = YOLO("best_new.pt")

results = model("thumb1.jpg", conf=0.3, classes=[0])
print(f"People detected: {len(results[0].boxes)}")
results[0].show()