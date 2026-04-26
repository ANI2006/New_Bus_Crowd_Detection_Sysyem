from ultralytics import YOLO

# Use the standard pretrained model - it knows "person" class from COCO
model = YOLO("best_new.pt")  # download automatically

results = model("thumb1.jpg", conf=0.3, classes=[0])  # class 0 = person in COCO
print(f"People detected: {len(results[0].boxes)}")
results[0].show()