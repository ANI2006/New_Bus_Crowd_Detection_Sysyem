# Training was done on Google Colab
# Model saved as best.pt
# Dataset: https://universe.roboflow.com/freelancing-4zdrw/people-in-bus

from roboflow import Roboflow
from ultralytics import YOLO

# Step 1: Download the dataset
rf = Roboflow(api_key="L0wCglKCTM4T8dXHKzgn")
project = rf.workspace("freelancing-4zdrw").project("people-in-bus")
dataset = project.version(4).download("yolov8")

# Step 2: Train the model
model = YOLO("yolov8n.pt")

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",        # uses your M2 GPU
    name="bus_model"
)

print("✅ Training complete! Model saved in runs/detect/bus_model/weights/best.pt")

#pca