# Training was done on Google Colab
# Dataset: https://universe.roboflow.com/freelancing-4zdrw/people-in-bus

from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="L0wCglKCTM4T8dXHKzgn")
project = rf.workspace("freelancing-4zdrw").project("people-in-bus")
dataset = project.version(4).download("yolov8")

model = YOLO("yolov8n.pt")

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",
    name="bus_model"
)

print("✅ Training complete! Model saved in runs/detect/bus_model/weights/best.pt")

#pca