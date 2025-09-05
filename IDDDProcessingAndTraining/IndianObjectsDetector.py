import os
from pathlib import Path
import cv2
import math
import torch
from ultralytics import YOLO

input_path = r"C:\Users\Xeron\Videos\IndianTrafficVideo.mp4"
model_path = r"C:\Users\Xeron\OneDrive\Documents\Programs\IDDDProcessingAndTraining\IDDDetectionsYOLODataset_TrainingOutput\train\weights\IDDDYOLO11m.pt"
output_dir = r"C:\Users\Xeron\Videos\NewOutput"

Path(output_dir).mkdir(parents=True, exist_ok=True)

cap_tmp = None
is_image = False
ext = os.path.splitext(input_path)[1].lower()
if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
    is_image = True
else:
    cap_tmp = cv2.VideoCapture(input_path)

if is_image:
    img = cv2.imread(input_path)
    if img is None:
        raise SystemExit("Could not read image")
    frame_height, frame_width = img.shape[:2]
    fps = 1
    total_frames = 1
else:
    if not cap_tmp.isOpened():
        raise SystemExit("Could not open video")
    frame_width = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_tmp.get(cv2.CAP_PROP_FPS) or 30)
    total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_tmp.release()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def class_color(name):
    h = abs(hash(name)) % (256 ** 3)
    r = (h >> 16) & 255
    g = (h >> 8) & 255
    b = h & 255
    if r + g + b < 100:
        r = 255 - r
        g = 255 - g
        b = 255 - b
    return (int(b), int(g), int(r))

model = YOLO(model_path)

def annotate_frame(frame, result, model_names):
    h, w = frame.shape[:2]
    out = frame.copy()
    if result is None:
        return out
    try:
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
    except Exception:
        return out
    for (x1, y1, x2, y2), cid in zip(xyxy, cls_ids):
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        cname = model_names.get(int(cid), str(int(cid)))
        color = class_color(cname)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, cname, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out

if is_image:
    results = model.predict(source=img, conf=0.3, device=device)
    result = results[0] if results else None
    annotated = annotate_frame(img, result, model.names if hasattr(model, "names") else {})
    base = Path(input_path).stem
    out_path = Path(output_dir) / f"{base}_pred.png"
    cv2.imwrite(str(out_path), annotated)
    print(str(out_path))
else:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = Path(output_dir) / "yolo_predictions.mp4"
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.3, device=device)
        result = results[0] if results else None
        annotated = annotate_frame(frame, result, model.names if hasattr(model, "names") else {})
        out.write(annotated)
        processed += 1
    cap.release()
    out.release()
    print(str(out_path))
    print(processed)
