# Import All the Required Libraries
import json
import math
import contextlib
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "data" / "gray1.jpg"
WEIGHTS_PATH = BASE_DIR / "weights" / "best.pt"
YOLOV10_DIR = BASE_DIR / "yolov10"
JSON_DIR = BASE_DIR / "json"
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = BASE_DIR / "licensePlatesDatabase.db"

# Class Names
className = ["License"]


def _ensure_bgr(frame):
    """Ensure frame is 3-channel BGR for YOLO/OpenCV drawing."""
    # YOLOv10 expects 3-channel images; some sources (e.g., PNG) decode as BGRA.
    if frame is None:
        return None
    if len(frame.shape) == 2:
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame



def paddle_ocr(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return ""

    frame = frame[y1:y2, x1:x2]
    result = ocr.predict(frame)
    text = ""
    best_score = 0
    if result and isinstance(result, list):
        r0 = result[0] if result else {}
        texts = r0.get("rec_texts", []) if isinstance(r0, dict) else []
        scores = r0.get("rec_scores", []) if isinstance(r0, dict) else []
        for t, s in zip(texts, scores):
            if s is None:
                s = 0
            else:
                try:
                    if isinstance(s, float) and math.isnan(s):
                        s = 0
                except Exception:
                    s = 0
            if s > best_score:
                best_score = s
                text = t
    if best_score * 100 < 60:
        text = ""
    pattern = re.compile(r'[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")
    return str(text)



def save_json(license_plates, startTime, endTime):
    #Generate individual JSON files for each 20-second interval
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plate": list(license_plates)
    }
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    interval_file_path = str(JSON_DIR / ("output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"))
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent = 2)

    #Cummulative JSON File
    cummulative_file_path = str(JSON_DIR / "LicensePlateData.json")
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    #Add new intervaal data to cummulative data
    existing_data.append(interval_data)

    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent = 2)

    #Save data to SQL database
    save_to_database(license_plates, startTime, endTime)



def save_to_database(license_plates, start_time, end_time):
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        '''
            CREATE TABLE IF NOT EXISTS LicensePlates(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT,
                end_time TEXT,
                license_plate TEXT
            )
        '''
    )
    for plate in license_plates:
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
    conn.commit()
    conn.close()



def main():
    import cv2

    # Prefer the local `yolov10/` folder (repo clone) over a globally installed package.
    # The YOLOv10 repo contains an `ultralytics` package; adding the repo root to sys.path
    # allows `from ultralytics import YOLOv10` to work without requiring a separate install.
    if YOLOV10_DIR.exists() and YOLOV10_DIR.is_dir():
        sys.path.insert(0, str(YOLOV10_DIR))
    else:
        raise FileNotFoundError(
            f"Missing required folder: {YOLOV10_DIR}\n"
            "Clone YOLOv10 into this project as a subfolder named 'yolov10'."
        )

    try:
        from ultralytics import YOLOv10
    except Exception as e:
        raise RuntimeError(
            "Failed to import YOLOv10 from the local 'yolov10' folder. "
            "Make sure the yolov10 repo is complete and its dependencies are installed."
        ) from e

    preview_window = "Live Preview"
    preview_enabled = True

    # PaddleOCR 3.x (via PaddleX) checks model hoster connectivity at import/init time.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from paddleocr import PaddleOCR

    global ocr

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {VIDEO_PATH}\n"
            "Put a video/image into the data/ folder or update VIDEO_PATH in main.py."
        )
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Weights not found: {WEIGHTS_PATH}\n"
            "Place your trained weights at weights/best.pt"
        )

    # Initialize OCR and model inside main() so importing this file doesn't run heavy side effects.
    ocr = PaddleOCR(use_textline_orientation=True)

    # PyTorch >= 2.6 defaults `torch.load(..., weights_only=True)` which can block
    # Ultralytics checkpoints unless the model class is allowlisted.
    # This keeps the safer default while allowing trusted YOLOv10 weights to load.
    try:
        import torch

        try:
            from ultralytics.nn.tasks import DetectionModel

            safe_globals_cm = torch.serialization.safe_globals([DetectionModel])
        except Exception:
            safe_globals_cm = contextlib.nullcontext()
    except Exception:
        safe_globals_cm = contextlib.nullcontext()

    with safe_globals_cm:
        model = YOLOv10(str(WEIGHTS_PATH))

    # Source can be either a video or a single image.
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    is_image = VIDEO_PATH.suffix.lower() in image_suffixes

    cap = None
    total_frames = 0
    fps = 30.0
    width = 0
    height = 0
    image_frame = None

    if is_image:
        image_frame = cv2.imread(str(VIDEO_PATH), cv2.IMREAD_UNCHANGED)
        if image_frame is None:
            raise RuntimeError(f"Failed to read image: {VIDEO_PATH}")
        image_frame = _ensure_bgr(image_frame)
        height, width = image_frame.shape[:2]
        fps = 1.0
        total_frames = 1
    else:
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        if not cap.isOpened():
            # Some OpenCV builds can't decode certain formats; fall back to image read.
            fallback = cv2.imread(str(VIDEO_PATH), cv2.IMREAD_UNCHANGED)
            if fallback is None:
                raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")
            image_frame = _ensure_bgr(fallback)
            height, width = image_frame.shape[:2]
            fps = 1.0
            total_frames = 1
            is_image = True
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    writer = None
    out_path = None
    output_label = ""
    if is_image:
        out_path = OUTPUT_DIR / f"processed_{VIDEO_PATH.stem}_{timestamp}.png"
        output_label = out_path.name
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = OUTPUT_DIR / f"processed_{VIDEO_PATH.stem}_{timestamp}.mp4"
        output_label = out_path.name
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open output writer: {out_path}")

    startTime = datetime.now()
    license_plates = set()
    count = 0

    try:
        last_frame = None
        if is_image:
            frames_iter = [(True, image_frame)]
        else:
            frames_iter = iter(lambda: cap.read(), (False, None))

        for ret, frame in frames_iter:
            if not ret or frame is None:
                break

            frame = _ensure_bgr(frame)

            currentTime = datetime.now()
            count += 1

            if total_frames > 0:
                pct = (count / total_frames) * 100
                sys.stdout.write(
                    f"\rProcessing & saving: {pct:6.2f}%  ({count}/{total_frames})  ->  {output_label}"
                )
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\rProcessing & saving: frame {count}  ->  {output_label}")
                sys.stdout.flush()

            results = model.predict(frame, conf=0.45, verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    label = paddle_ocr(frame, x1, y1, x2, y2)
                    if label:
                        license_plates.add(label)

                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

            if (currentTime - startTime).seconds >= 20:
                endTime = currentTime
                save_json(license_plates, startTime, endTime)
                startTime = currentTime
                license_plates.clear()

            # Live preview while still processing + saving.
            # Controls:
            # - Press 'c' to close the preview window and keep processing.
            # - Press 'q' to quit processing early.
            if preview_enabled:
                cv2.imshow(preview_window, frame)

                # If the user clicked the X button, disable preview and continue.
                try:
                    if cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) < 1:
                        preview_enabled = False
                except Exception:
                    preview_enabled = False

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    preview_enabled = False
                    try:
                        cv2.destroyWindow(preview_window)
                    except Exception:
                        pass
                elif key == ord('q'):
                    break

            last_frame = frame
            if writer is not None:
                writer.write(frame)
    finally:
        # Ensure we finish the progress line
        if count > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Flush remaining plates at end of video
        if license_plates:
            save_json(license_plates, startTime, datetime.now())

        # If the source was a single image, save the final annotated frame.
        if is_image and last_frame is not None and out_path is not None:
            ok = cv2.imwrite(str(out_path), last_frame)
            if ok:
                sys.stdout.write(f"Saved image: {out_path}\n")
                sys.stdout.flush()

        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
