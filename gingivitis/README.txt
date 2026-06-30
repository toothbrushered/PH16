To train, val and test, activate and go to your conda environment (in this case, hcjc-gingivitis) and paste this:

train: yolo detect train data=/path/to/dental.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 project=runs/dental name=exp1
val :  yolo detect val data=/path/to/dental.yaml model=runs/dental/exp1/weights/best.pt
test : yolo detect val data=/path/to/dental.yaml model=runs/dental/exp1/weights/best.pt split=test

Replace /path/to/dental.yaml with /mnt/2tb-samsung/zychin/HCJC2026/dental.yaml
