## Demo Video: https://drive.google.com/file/d/1XnvV-ZSvd7zdES0NtLIx4ATXq9iksnxh/view?usp=sharing

# License Plate Detection + OCR (YOLOv10 + EasyOCR)

## About this project

This repository is the core of a Vehicle License Plate Recognition workflow that can be used as the backend for a parking system.
It processes a video (CCTV/phone recording) or a single image, detects license plates with a YOLOv10 model (`best.pt`), reads the
plate text using **EasyOCR** (supports **English + Bangla**), and stores the recognized plates with timestamps.

### Features

- **Dual Language OCR**: Supports both English and Bangla (Bengali) license plates
- **Bangla → English Transliteration**: Bangla text is automatically converted to English (e.g., `ঢাকা মেট্রো` → `Dhaka Metro`)
- **GPU Acceleration**: Auto-detects NVIDIA GPU (CUDA) for faster OCR, falls back to CPU
- **Live Preview**: Resizable preview window while processing
- **Multiple Output Formats**: JSON, SQLite database, and annotated video/image

### Typical use cases:

- Parking entry/exit logging (store plate + time range)
- Traffic/camera footage indexing by plate number
- Basic audit trail using a local SQLite database

### Outputs:

- An annotated output video/image in `output/`
- JSON in `json/` (per-interval + cumulative)
- SQLite database `licensePlatesDatabase.db`

## How to run (Windows)

### 0) Prerequisites

- Python 3.11 (recommended)
- Git
- NVIDIA GPU with CUDA (optional, for faster OCR)

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

| Key | Action |
|-----|--------|
| `c` | Close preview window (processing continues) |
| `q` | Stop processing early |
| Any key | Close preview (for images only) |

The preview window is **resizable** - drag edges/corners to resize.

## Supported License Plates

| Language | Example Input | Output |
|----------|---------------|--------|
| English | `LS15 EBC` | `LS15 EBC` |
| Bangla | `ঢাকা মেট্রো-গ ২০-২৮৬৪` | `Dhaka Metro-Ga 20-2864` |

## Optional: Tesseract Fallback

For improved accuracy on difficult plates, install Tesseract OCR:

1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. During install, select **Bengali** and **English** language packs
3. Add Tesseract to PATH

The system will automatically use Tesseract as a fallback when EasyOCR confidence is low.

## Notebook (training)

This repo also includes a training notebook used on Kaggle to train the custom YOLOv10 model that produced `best.pt`.

- Notebook path: `notebook/license-plate-detection-yolov10-custom-model-train.ipynb`
- It covers: cloning YOLOv10, installing dependencies, downloading a dataset, training, and running sample predictions.

## View SQLite Database

Upload `licensePlatesDatabase.db` on https://inloop.github.io/sqlite-viewer/
