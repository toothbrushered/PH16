To train, val and test, activate and go to your conda environment (in this case, hcjc-gingivitis) and paste this:

# 1. Install dependencies
pip install albumentations opencv-python ultralytics --break-system-packages

# 2. Run augmentation
python3 augment_all.py

# 3. Train
yolo detect train data=/mnt/2tb-samsung/zychin/HCJC2026/Gingivitis_Dataset/dental.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 project=runs/dental name=exp1

# 4. Validate
yolo detect val data=/mnt/2tb-samsung/zychin/HCJC2026/Gingivitis_Dataset/dental.yaml model=runs/dental/exp1/weights/best.pt

# 5. Test
yolo detect val data=/mnt/2tb-samsung/zychin/HCJC2026/Gingivitis_Dataset/dental.yaml model=runs/dental/exp1/weights/best.pt split=test

Replace /path/to/dental.yaml with /mnt/2tb-samsung/zychin/HCJC2026/dental.yaml
