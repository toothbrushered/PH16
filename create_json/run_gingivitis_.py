import json
from ultralytics import YOLO
import os

GINGIVITIS_MODEL_PATH = '/mnt/2tb-samsung/zychin/HCJC2026/Gingivitis_Dataset/yolo26n.pt'
image_paths = ["path/to/photo1.jpg", "path/to/photo2.jpg"]#these are just placeholders, replace w actual iamge

#same structure as malocclusion but w extra logic
model = YOLO(GINGIVITIS_MODEL_PATH)
results = model.predict(image_paths, conf=0.25)

JAW_LABELS = {"upper_jaw": "maxilla", "lower_jaw": "mandible"} #maps YOLO class names to dental terms
SEVERITY_SCORE = {
    "healthy": 0,
    "gingivitis_1": 1,
    "gingivitis_2": 2,
    "gingivitis_3": 3,
    "gingivitis_4": 4,
} #maps gingivitis severity to follow MGI index

jaw_boxes = [] #jaw region labels
severity_boxes = [] #severity labels

for item in results:
    for box in item.boxes:
        cls_idx = int(box.cls.item())
        label = item.names[cls_idx]
        bbox = box.xyxy[0].tolist() #bounding box coordinates format (x_min, y_min, x_max, y_max)

        if label in JAW_LABELS:
            jaw_boxes.append({"label": label, "bbox": bbox})
        elif label in SEVERITY_SCORE:
            severity_boxes.append({"label": label, "bbox": bbox})

def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2) #finds middle point of box
    
mgi_scores = {"maxilla": {"overall": 0}, "mandible": {"overall": 0}}

for sev in severity_boxes:
    cx, cy = get_center(sev["bbox"])
    for jaw in jaw_boxes:
        x1, y1, x2, y2 = jaw["bbox"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            jaw_name = JAW_LABELS[jaw["label"]]
            score = SEVERITY_SCORE[sev["label"]]
            if score > mgi_scores[jaw_name]["overall"]:
                mgi_scores[jaw_name]["overall"] = score
                #for each severity detection, check every jaw box. if yes, we know which jaw the severity belongs to and keep the highest severity seen for that jaw

output = {"mgi_scores": mgi_scores}

os.makedirs("outputs", exist_ok=True)
with open("outputs/gingivitis_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved:", output)
