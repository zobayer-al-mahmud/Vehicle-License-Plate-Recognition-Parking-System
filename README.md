# License Plate Detection + OCR (YOLOv10 + PaddleOCR)

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

- Put your model weights at `weights/best.pt`
- Put your input video/image under `data/`
- Update `VIDEO_PATH` in `main.py` to point to your file

### 5) Run

```powershell
python main.py
```

You can also run without activating the venv:

```powershell
.\.venv\Scripts\python.exe .\main.py
```

## Live preview controls

- Press `c` to close the live preview window (processing continues)
- Press `q` to stop processing early


