import json
from ultralytics import YOLO
import os

GINGIVITIS_MODEL_PATH = '/mnt/2tb-samsung/zychin/HCJC2026/Gingivitis_Dataset/yolo26n.pt'
image_paths = ["path/to/photo1.jpg", "path/to/photo2.jpg"]#these are just placeholders, replace w actual iamge

model = YOLO(GINGIVITIS_MODEL_PATH)
results = model.predict(image_paths, conf=0.25)

JAW_LABELS = {"upper_jaw": "maxilla", "lower_jaw": "mandible"}
SEVERITY_SCORE = {
    "healthy": 0,
    "gingivitis_1": 1,
    "gingivitis_2": 2,
    "gingivitis_3": 3,
    "gingivitis_4": 4,
}

jaw_boxes = []
severity_boxes = []

for item in results:
    for box in item.boxes:
        cls_idx = int(box.cls.item())
        label = item.names[cls_idx]
        bbox = box.xyxy[0] #undersatnd waht this does

        if label in JAW_LABELS:
            jaw_boxes.append({"label": label, "bbox": bbox})
        elif label in SEVERITY_SCORE:
            severity_boxes.append({"label": label, "bbox": bbox})
  
output = {"detected_labels": sorted(all_labels)}

os.makedirs("outputs", exist_ok=True)
with open("outputs/gingivitis_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved:", output)
