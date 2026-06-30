# from pathlib import Path
# import shutil
# import random

# SOURCE = Path("archive/PetImages")
# OUTPUT = Path("dataset")

# random.seed(42)

# for cls in ["Cat", "Dog"]:

#     images = list((SOURCE / cls).glob("*.jpg"))

#     # Remove corrupted files
#     valid_images = []
#     for img in images:
#         try:
#             if img.stat().st_size > 0:
#                 valid_images.append(img)
#         except:
#             pass

#     random.shuffle(valid_images)

#     n = len(valid_images)

#     train_end = int(0.8 * n)
#     val_end = int(0.9 * n)

#     splits = {
#         "train": valid_images[:train_end],
#         "val": valid_images[train_end:val_end],
#         "test": valid_images[val_end:]
#     }

#     for split, files in splits.items():

#         dst = OUTPUT / split / cls.lower()
#         dst.mkdir(parents=True, exist_ok=True)

#         for file in files:
#             shutil.copy(file, dst / file.name)

# print("Done!")
import os
from pathlib import Path
from ultralytics import YOLO
# =====================================================
# STEP 2: TRAIN MODEL
# =====================================================

print("Training model...")

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset",
    epochs=20,
    imgsz=224,
    batch=32,
    device=0,      # Use GPU
    workers=8,
    project="cats_vs_dogs",
    name="run1"
)

# =====================================================
# STEP 3: VALIDATION SCORE
# =====================================================

print("\nEvaluating model...")

metrics = model.val()

print(metrics)

# =====================================================
# STEP 4: TEST IMAGES
# =====================================================

best_model = YOLO(
    "runs/classify/cats_vs_dogs/run1/weights/best.pt"
)

test_folder = Path("dataset/test")

correct = 0
total = 0

for class_folder in test_folder.iterdir():

    true_label = class_folder.name

    for img in class_folder.glob("*.jpg"):

        result = best_model.predict(
            source=str(img),
            verbose=False
        )[0]

        predicted_class = result.names[result.probs.top1]

        if predicted_class == true_label:
            correct += 1

        total += 1

accuracy = 100 * correct / total

print("\n===================================")
print(f"Test Accuracy: {accuracy:.2f}%")
print("===================================")
