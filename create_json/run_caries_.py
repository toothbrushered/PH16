import json
from ultralytics import YOLO
import os

CARIES_MODEL_PATH = '/mnt/2tb-samsung/zychin/HCJC2026/Caries_Dataset/yolov26n.pt'
image_paths = ["path/to/photo1.jpg", "path/to/photo2.jpg"]#these are just placeholders, replace w actual iamge

model = YOLO(CARIES_MODEL_PATH)
results = model.predict(image_paths, conf=0.25)

TOOTH_TYPE = {
    "caries_permanent": "permanent",
    "caries_primary": "primary",
}

teeth = []
for item in results:
    for box in item.boxes:
        cls_idx = int(box.cls.item())
        label = item.names[cls_idx]
        tooth_type = TOOTH_TYPE[label].get()
        teeth.append({"type": tooth_type, "caries": True})

output = {"teeth": teeth}

os.makedirs("outputs", exist_ok=True)
with open("outputs/caries_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved:", output)

