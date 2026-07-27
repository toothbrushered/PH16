import json
from ultralytics import YOLO
import os

MALOCCLUSION_MODEL_PATH = '/mnt/2tb-samsung/zychin/HCJC2026/teeth/runs/detect/omni_run/weights/best.pt'
image_paths = ["path/to/photo1.jpg", "path/to/photo2.jpg"]#these are just placeholders, replace w actual iamge

model = YOLO(MALOCCLUSION_MODEL_PATH)
results = model.predict(image_paths, conf=0.25)

all_labels = set()
for item in results:
    for box in item.boxes:
        cls_idx = int(box.cls.item())
        label = item.names[cls_idx]
        all_labels.add(label)

output = {"detected_labels": sorted(all_labels)}

os.makedirs("outputs", exist_ok=True)
with open("outputs/malocclusion_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved:", output)