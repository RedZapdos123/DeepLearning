import os
import cv2
import math
import multiprocessing
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

PATCH_AREA_PERCENTAGE = 10
video_path = r"C:\Users\Xeron\Videos\PrayagIntersection\PrayagIntersection1.mp4"
model_path = r"C:\Users\Xeron\OneDrive\Documents\Programs\RoadVehiclesYOLODatasetProTraining\RoadVehiclesYOLO11m.pt"
output_project = r"C:\Users\Xeron\Videos\NewOutput"

cap_tmp = cv2.VideoCapture(video_path)
frame_width = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap_tmp.get(cv2.CAP_PROP_FPS) or 30)
total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
cap_tmp.release()

frame_area = frame_width * frame_height
patch_area = (PATCH_AREA_PERCENTAGE / 100) * frame_area
patch_size = max(64, int(math.sqrt(patch_area)))
slice_height = patch_size
slice_width = patch_size

_detection_model = None
_slice_height = None
_slice_width = None
_overlap_height_ratio = None
_overlap_width_ratio = None
_postprocess_type = None
_postprocess_match_threshold = None
_postprocess_class_agnostic = None

def init_worker(model_path_arg, confidence_threshold, device, slice_h, slice_w, overlap_h, overlap_w, postprocess_type, postprocess_match_threshold, postprocess_class_agnostic, image_size):
    global _detection_model, _slice_height, _slice_width, _overlap_height_ratio, _overlap_width_ratio, _postprocess_type, _postprocess_match_threshold, _postprocess_class_agnostic
    _detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path_arg,
        confidence_threshold=confidence_threshold,
        device=device,
        image_size=image_size
    )
    _slice_height = slice_h
    _slice_width = slice_w
    _overlap_height_ratio = overlap_h
    _overlap_width_ratio = overlap_w
    _postprocess_type = postprocess_type
    _postprocess_match_threshold = postprocess_match_threshold
    _postprocess_class_agnostic = postprocess_class_agnostic

def process_frame(args):
    idx, frame = args
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = get_sliced_prediction(
        frame_rgb,
        _detection_model,
        slice_height=_slice_height,
        slice_width=_slice_width,
        overlap_height_ratio=_overlap_height_ratio,
        overlap_width_ratio=_overlap_width_ratio,
        perform_standard_pred=True,
        postprocess_type=_postprocess_type,
        postprocess_match_threshold=_postprocess_match_threshold,
        postprocess_class_agnostic=_postprocess_class_agnostic
    )
    annotated_frame = frame.copy()
    if hasattr(result, "object_prediction_list") and result.object_prediction_list:
        for prediction in result.object_prediction_list:
            bbox = prediction.bbox
            x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
            confidence = float(prediction.score.value)
            class_name = prediction.category.name
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (max(0, x1), max(12, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return idx, annotated_frame

if __name__ == "__main__":
    Path(output_project).mkdir(parents=True, exist_ok=True)
    Path(f"{output_project}/Predictions").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = f"{output_project}/Predictions/sahi_predictions.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    cpu_count = multiprocessing.cpu_count()
    processes = max(1, cpu_count - 1)
    chunk_size = processes * 2

    device = "cpu"
    image_size = max(frame_width, frame_height)

    pool = multiprocessing.Pool(
        processes=processes,
        initializer=init_worker,
        initargs=(model_path, 0.3, device, slice_height, slice_width, 0.2, 0.2, "NMS", 0.5, True, image_size)
    )

    frame_index = 0
    processed_frames = 0
    try:
        while True:
            batch = []
            for _ in range(chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch.append((frame_index, frame.copy()))
                frame_index += 1
            if not batch:
                break
            results = pool.map(process_frame, batch)
            results.sort(key=lambda x: x[0])
            for idx, annotated in results:
                out.write(annotated)
                processed_frames += 1
    finally:
        pool.close()
        pool.join()
        cap.release()
        out.release()

    print(out_path)
    print(f"{frame_width}x{frame_height}")
    print(f"{patch_size}x{patch_size}")
    print(processed_frames)
