## Demo Video: https://drive.google.com/file/d/1XnvV-ZSvd7zdES0NtLIx4ATXq9iksnxh/view?usp=sharing

# License Plate Detection + OCR (YOLOv10 + PaddleOCR)

## About this project

This repository is the core of a Vehicle License Plate Recognition workflow that can be used as the backend for a parking system.
It processes a video (CCTV/phone recording) or a single image, detects license plates with a YOLOv10 model (`best.pt`), reads the
plate text using PaddleOCR, and stores the recognized plates with timestamps.

Typical use cases:

- Parking entry/exit logging (store plate + time range)
- Traffic/camera footage indexing by plate number
- Basic audit trail using a local SQLite database

Detects license plates in a video/image using YOLOv10, reads plate text with PaddleOCR, and saves results to:

- An annotated output video/image in `output/`
- JSON in `json/` (per-interval + cumulative)
- SQLite database `licensePlatesDatabase.db`

## How to run (Windows)

### 0) Prerequisites

- Python 3.11 (recommended)
- Git

### 1) Clone YOLOv10 into this project

From the project root:

```powershell
git clone https://github.com/THU-MIG/yolov10.git yolov10
```

### 2) Create + activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

### 4) Provide inputs

- Put your model weights at `weights/best.pt` (custom-trained by me on Kaggle)
- Put your input video/image under `data/`
- Update `VIDEO_PATH` in `main.py` to point to your file

### 5) Run

```powershell
sqldb.py
python main.py
```

You can also run without activating the venv:

```powershell
.\.venv\Scripts\python.exe .\main.py
```

## Live preview controls

- Press `c` to close the live preview window (processing continues)
- Press `q` to stop processing early

## Notebook (training)

This repo also includes a training notebook used on Kaggle to train the custom YOLOv10 model that produced `best.pt`.

- Notebook path: `notebook/license-plate-detection-yolov10-custom-model-train.ipynb`
- It covers: cloning YOLOv10, installing dependencies, downloading a dataset, training, and running sample predictions.

upload licensePlatesDatabase.db on https://inloop.github.io/sqlite-viewer/
