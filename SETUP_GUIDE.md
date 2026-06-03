# Running OMNI Malocclusion Detection with MMDetection — Setup & Reference Guide

A step-by-step record of how I set up the [OMNI](https://github.com/RoundFaceJ/OMNI) deep-learning baselines on a local Windows machine with an NVIDIA GPU, and ran inference with a trained Faster R-CNN model to visualize malocclusion detections on oral images.

This guide is written so I (or anyone) can reproduce the environment from scratch.

---

## 1. What this is

The OMNI project benchmarks six object-detection models on the OMNI dataset (4,166 oral RGB images for automated malocclusion assessment). Five of them — Faster R-CNN, Mask R-CNN, EfficientDet, DETR, and Deformable DETR — are built on the [MMDetection](https://github.com/open-mmlab/mmdetection) framework (v3.0.0). The sixth is a separate GNN model (GraphTeeth).

The OMNI repo does **not** bundle a full copy of MMDetection. Its `mmdetection/` folder contains only the team's *additions* — custom dataset code, evaluation metrics, configuration files (`my_configs/`), and training/testing scripts. These are meant to be laid on top of an official MMDetection v3.0.0 install. That detail drives the whole setup below.

---

## 2. System used

- Windows, with an NVIDIA GPU (driver supporting up to CUDA 13.2)
- Miniforge (conda) for environment management
- Conda environment named `openmmlab`, Python **3.8.20**

> Note on versions: the GPU driver's "CUDA 13.2" is only the maximum it supports. PyTorch CUDA builds are backward-compatible, so a lower CUDA build (12.1) runs fine. The real constraint is **MMCV**, whose prebuilt wheels lag behind the newest PyTorch. The combination below was chosen specifically so every package installs from a prebuilt wheel — no compiling from source on Windows.

The tested-good combination:

| Component | Version |
|-----------|---------|
| Python | 3.8.20 |
| PyTorch | 2.1.0 + cu121 |
| torchvision | 0.16.0 + cu121 |
| MMEngine | 0.10.7 |
| MMCV | 2.1.0 |
| MMDetection | 3.0.0 (from source) |

---

## 3. Setup steps

All commands are run inside the activated conda environment (the prompt shows `(openmmlab)`).

### 3.1 Install PyTorch (GPU build)

```
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible to PyTorch:

```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected output: `2.1.0+cu121 12.1 True`. The `True` is essential — if it prints `False`, stop and fix the GPU/driver setup before continuing.

### 3.2 Install MMEngine and MMCV via mim

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
```

`mim` resolves the correct prebuilt MMCV wheel for the installed PyTorch+CUDA combo. On the successful run it downloaded `mmcv-2.1.0-cp38-cp38-win_amd64.whl` (a `.whl`, not a `.tar.gz` — that confirms no source build was triggered).

### 3.3 Clone the OMNI repo

```
git clone https://github.com/RoundFaceJ/OMNI.git
cd OMNI
```

### 3.4 Lay official MMDetection v3.0.0 under the OMNI overlay

The repo's `mmdetection/` folder has no `setup.py`, so it can't be installed directly. Clone the official framework into a temporary folder, then copy it in *without overwriting* the OMNI custom files (the `/XN /XO /XC` flags skip files that already exist so OMNI's versions win):

```
git clone -b v3.0.0 https://github.com/open-mmlab/mmdetection.git mmdetection_official
robocopy mmdetection_official mmdetection /E /XC /XN /XO /XD .git
```

(robocopy prints a summary and exits; an exit code of 1 means "files copied," not an error.)

### 3.5 Install MMDetection in editable mode

```
cd mmdetection
pip install -v -e .
```

### 3.6 Fix the MMCV version guard

MMDetection 3.0.0 hard-caps MMCV at `< 2.1.0` and will throw `AssertionError: MMCV==2.x.0 is used but incompatible` on import. MMCV 2.1.0 works fine in practice with these standard detectors, so bump the guard. Open the init file:

```
notepad mmdet\__init__.py
```

Change:

```python
mmcv_maximum_version = '2.1.0'
```

to:

```python
mmcv_maximum_version = '2.2.0'
```

Save and close.

### 3.7 Verify the install

```
python -c "import mmdet; print(mmdet.__version__)"
```

Expected output: `3.0.0` with no assertion error.

### 3.8 Clean up the temporary clone

```
cd ..
rmdir /s /q mmdetection_official
```

---

## 4. Confirm the pipeline works (sanity check)

Before touching the OMNI data, confirm the whole chain runs on a standard model and the bundled demo image. From inside the `mmdetection` folder:

```
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest checkpoints
python demo/image_demo.py demo/demo.jpg checkpoints/rtmdet_tiny_8xb32-300e_coco.py --weights checkpoints/<the_downloaded_weights>.pth --device cuda:0
```

The result is saved to `outputs/vis/`. If `outputs\vis\demo.jpg` shows boxes on the cars/people, the GPU + PyTorch + MMCV + MMDetection chain is fully working.

> The console prints a few harmless warnings every run: `Failed to search registry with scope "mmdet"`, `Failed to add LocalVisBackend`, a state-dict key mismatch, and a `torch.meshgrid` deprecation notice. None affect the output. The line that matters is `results have been saved at outputs`.

---

## 5. Add the OMNI dataset and trained weights

Both are downloaded from links in the repo READMEs (Google Drive).

### 5.1 Trained model weights

Download `work_dirs3` (trained baselines for all five models) and place it inside the `mmdetection` folder:

```
mmdetection/work_dirs3/
    faster-rcnn/
        faster-rcnn_r50_fpn_100_omni/
            best_coco_bbox_mAP_50_epoch_16.pth
        ... (r101, RoIPool variants)
    mask-rcnn/  detr/  deformable_detr/  efficientdet/  ...
```

### 5.2 The dataset

Download and unzip the OMNI_COCO dataset to `mmdetection/data/OMNI_COCO/`. Final structure:

```
mmdetection/data/OMNI_COCO/
    annotations/   (instances_train.json, instances_val.json, instances_test.json)
    train/
    val/
    test/
```

> The image folders are nested several levels deep: `test/<patient_id>/<date>/<imagefile>.JPG`. Each leaf folder holds up to five `.JPG` files (suffixes `1`–`5`) corresponding to the five camera views. Note the extension is uppercase `.JPG`.

To list every test image and its full path:

```
dir /s /b data\OMNI_COCO\test\*.JPG
```

---

## 6. Run inference and visualize detections

Use the official demo script with a trained OMNI checkpoint instead of the standard model. Pick a real image path from the listing above.

```
python demo/image_demo.py data/OMNI_COCO/test/2018.04/2018.07.10/201804201807101.JPG my_configs/faster_rcnn/faster-rcnn_r50_fpn_100_omni.py --weights work_dirs3/faster-rcnn/faster-rcnn_r50_fpn_100_omni/best_coco_bbox_mAP_50_epoch_16.pth --device cuda:0 --pred-score-thr 0.3
```

Arguments:
- **first positional** — the input image path (forward slashes; quote it if it contains spaces)
- **second positional** — the model config
- **`--weights`** — the trained `.pth` checkpoint
- **`--device cuda:0`** — run on the GPU
- **`--pred-score-thr 0.3`** — minimum confidence to draw a box (lower = more boxes, e.g. `0.2`; higher = only confident ones, e.g. `0.5`)

The annotated image is written under `outputs/vis/`. Because the input was in a subfolder, find the exact output path with:

```
dir /s /b outputs\vis
```

Then open it:

```
start outputs\vis\201804201807101.JPG
```

### Example result

The output shows bounding boxes around individual teeth, each with a class label (e.g. HT, MR, SHT) and a confidence score:

![Sample detection](sample_detection.jpg)

---

## 7. Common mistakes & fixes (things that actually tripped me up)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `pip not recognised` | pip not in the fresh conda env | `python -m pip ...`, or `conda install pip` |
| `does not appear to be a Python project: neither setup.py nor pyproject.toml found` | The OMNI `mmdetection/` folder is an overlay, not the full framework | Lay official v3.0.0 underneath it (Section 3.4) |
| `AssertionError: MMCV==2.x.0 is used but incompatible` | MMDetection 3.0.0's MMCV cap | Bump `mmcv_maximum_version` (Section 3.6) |
| `No matching distribution found for puccinialin` (installing Jupyter `notebook`) | Newest `pywinpty` has no Python 3.8 wheel and needs a Rust build | `conda install -c conda-forge notebook`, or `pip install "pywinpty==2.0.10"` first |
| `FileNotFoundError: ... .jpg` during inference | Wrong/guessed image filename (model loads fine; only the file open fails) | Copy an exact path from `dir /s /b data\OMNI_COCO\test\*.JPG`; mind the uppercase `.JPG` |
| Two commands run as one / "cannot find the file specified" | Pasted two lines with no line break | Run each command on its own line |

---

## 8. Quick reference — full command sequence

```
:: 1. PyTorch (GPU)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"

:: 2. MMEngine + MMCV
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"

:: 3. OMNI repo + official mmdet overlay
git clone https://github.com/RoundFaceJ/OMNI.git
cd OMNI
git clone -b v3.0.0 https://github.com/open-mmlab/mmdetection.git mmdetection_official
robocopy mmdetection_official mmdetection /E /XC /XN /XO /XD .git
cd mmdetection
pip install -v -e .

:: 4. Edit mmdet\__init__.py: mmcv_maximum_version = '2.2.0'

:: 5. Verify
python -c "import mmdet; print(mmdet.__version__)"
cd ..
rmdir /s /q mmdetection_official

:: 6. Place work_dirs3/ and data/OMNI_COCO/ inside mmdetection/, then:
cd mmdetection
python demo/image_demo.py data/OMNI_COCO/test/<patient>/<date>/<image>.JPG my_configs/faster_rcnn/faster-rcnn_r50_fpn_100_omni.py --weights work_dirs3/faster-rcnn/faster-rcnn_r50_fpn_100_omni/best_coco_bbox_mAP_50_epoch_16.pth --device cuda:0 --pred-score-thr 0.3
```

---

## 9. Credits

- OMNI dataset, baselines, and trained weights: [RoundFaceJ/OMNI](https://github.com/RoundFaceJ/OMNI) (Xue et al., 2025, arXiv:2505.15637)
- Detection framework: [OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection)
