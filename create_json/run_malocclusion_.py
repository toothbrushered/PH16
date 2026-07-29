import json
from ultralytics import YOLO
import os

MALOCCLUSION_MODEL_PATH = '/mnt/2tb-samsung/zychin/HCJC2026/teeth/runs/detect/omni_run/weights/best.pt'
image_paths = ["path/to/photo1.jpg", "path/to/photo2.jpg"]#these are just placeholders, replace w actual iamge

model = YOLO(MALOCCLUSION_MODEL_PATH) #call the model that has the best weights
results = model.predict(image_paths, conf=0.25) #runs the model on the list of image | only detections w >25% confidence are kept

all_labels = set() #using a set means deuplicates are automatically removed
for item in results:
    for box in item.boxes:
        cls_idx = int(box.cls.item()) #gets class index
        label = item.names[cls_idx] #looks up human readable label names
        all_labels.add(label)

output = {"detected_labels": sorted(all_labels)}

os.makedirs("outputs", exist_ok=True)
with open("outputs/malocclusion_output.json", "w") as f:
    json.dump(output, f, indent=2) #save the unique detected labels as a sorted list. slm script reads this json file

print("Saved:", output)
