import cv2
import albumentations as A
import os
import glob

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.Rotate(limit=10, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

img_dir = '/mnt/2tb-samsung/zychin/HCJC2026/Gingivitis_Dataset/images/train'
lbl_dir = '/mnt/2tb-samsung/zychin/HCJC2026/Gingivitis_Dataset/labels/train'

all_images = []
for lbl_file in glob.glob(f'{lbl_dir}/*.txt'):
    if '_aug' in lbl_file:
        continue
    with open(lbl_file) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        continue
    img_file = lbl_file.replace('labels', 'images').replace('.txt', '.jpg')
    if os.path.exists(img_file):
        all_images.append((img_file, lbl_file))

print(f'Found {len(all_images)} original training images — augmenting 3x each')

count = 0
for idx, (img_path, lbl_path) in enumerate(all_images):
    # resize to 640 first to speed up processing
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with open(lbl_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.split()
        class_labels.append(parts[0])
        bboxes.append([float(x) for x in parts[1:]])

    for i in range(3):
        try:
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            if not augmented['bboxes']:
                continue
            aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

            base = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(f'{img_dir}/{base}_aug{i}.jpg', aug_img)
            with open(f'{lbl_dir}/{base}_aug{i}.txt', 'w') as f:
                for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                    f.write(f'{cls} {" ".join(map(str, bbox))}\n')
            count += 1
        except Exception as e:
            print(f'Skipping aug {i}: {e}')

    if idx % 100 == 0:
        print(f'Progress: {idx}/{len(all_images)}')

print(f'Done! Generated {count} augmented images')
