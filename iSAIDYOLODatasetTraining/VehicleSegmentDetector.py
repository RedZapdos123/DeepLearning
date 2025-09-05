import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

input_path = r"C:\Users\Xeron\Videos\PrayagIntersection\ForeignRoundabout.mp4"
model_path = r"C:\Users\Xeron\OneDrive\Documents\Programs\iSAIDYOLODatasetTraining\iSAIDYolo11n-SegDataset_TrainingOutput\train\weights\iSAIDYolo11n-Seg.pt"
output_dir = r"C:\Users\Xeron\Videos\NewOutput_Seg"

Path(output_dir).mkdir(parents=True, exist_ok=True)

cap_tmp = None
is_image = False
ext = os.path.splitext(input_path)[1].lower()
if ext in [".jpg", ".jpeg"]:
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

raw_names = model.names if hasattr(model, "names") else {}
if isinstance(raw_names, dict):
    model_names = {int(k): v for k, v in raw_names.items()}
elif isinstance(raw_names, (list, tuple)):
    model_names = {i: n for i, n in enumerate(raw_names)}
else:
    model_names = {}

vehicle_name_set = {"Large_Vehicle", "Small_Vehicle"}
allowed_class_ids = [cid for cid, nm in model_names.items() if nm in vehicle_name_set]
if len(allowed_class_ids) == 0:
    allowed_class_ids = []

def masks_from_result(result, frame_size):
    masks_list = []
    if result is None:
        return masks_list
    try:
        if hasattr(result, "masks") and result.masks is not None:
            mobj = result.masks
            if hasattr(mobj, "data") and mobj.data is not None:
                arr = mobj.data.cpu().numpy()
                for a in arr:
                    masks_list.append(a)
            elif hasattr(mobj, "xy") and mobj.xy is not None:
                polys = mobj.xy
                h, w = frame_size
                for poly in polys:
                    mask = np.zeros((h, w), dtype='uint8')
                    for p in poly:
                        pts = np.array(p, dtype=np.int32)
                        if pts.size == 0:
                            continue
                        cv2.fillPoly(mask, [pts], 1)
                    masks_list.append(mask)
    except Exception:
        pass
    return masks_list

def composite_and_annotate(frame, result, model_names, alpha=0.25):
    h, w = frame.shape[:2]
    frame_f = frame.astype(np.float32)
    overlay_color = np.zeros_like(frame_f, dtype=np.float32)
    alpha_acc = np.zeros((h, w), dtype=np.float32)
    masks = masks_from_result(result, (h, w))
    cls_ids = []
    if hasattr(result, "boxes") and result.boxes is not None:
        try:
            boxes = result.boxes
            if hasattr(boxes, "cls"):
                cls_ids = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            cls_ids = []
    for i, m in enumerate(masks):
        try:
            mask = m
            if mask.dtype != np.uint8 and mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            if mask.ndim == 3:
                mask = mask[0]
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask > 127).astype(np.uint8)
            if mask_bin.sum() == 0:
                continue
            cid = int(cls_ids[i]) if i < len(cls_ids) else -1
            if cid not in allowed_class_ids:
                continue
            cname = model_names.get(cid, str(cid))
            color = class_color(cname)
            color_arr = np.array(color, dtype=np.float32)
            mask_f = mask_bin.astype(np.float32) * alpha
            overlay_color += mask_f[..., None] * color_arr
            alpha_acc += mask_f
        except Exception:
            continue
    alpha_acc = np.clip(alpha_acc, 0.0, 1.0)
    composite = frame_f * (1.0 - alpha_acc[..., None]) + overlay_color
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    return composite

if is_image:
    results = model.predict(source=img, conf=0.3, device=device, classes=allowed_class_ids, save=False)
    result = results[0] if results else None
    annotated = composite_and_annotate(img, result, model_names if model_names else {})
    base = Path(input_path).stem
    out_path = Path(output_dir) / f"{base}_pred_seg.png"
    cv2.imwrite(str(out_path), annotated)
    print(str(out_path))
else:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = Path(output_dir) / "yolo_seg_predictions2.mp4"
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_width, frame_height))
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.3, device=device, classes=allowed_class_ids, save=False)
        result = results[0] if results else None
        annotated = composite_and_annotate(frame, result, model_names if model_names else {}, alpha=0.25)
        out.write(annotated)
        processed += 1
    cap.release()
    out.release()
    print(str(out_path))
    print(processed)
