#This program makes use of the RoboFlow trackers library (https://github.com/roboflow/trackers/blob/main/docs/index.md), 
#ultralytics' YOLOv11m-obb pretrained model for vehicle detections.
#Normal SORT tracker is being used for path tracking.

import supervision as sv
from trackers import SORTTracker
from ultralytics import YOLO
import numpy as np
import cv2

#Loading the yolo11m-oob pretrained model for vehicles detection in top-down aerial view.
model = YOLO(r"C:\Users\Xeron\Videos\PrayagIntersection\yolo11m-obb.pt")
#Loading the Normal SORT Tracker.
tracker = SORTTracker()

#Defining the OBB annotator, and the trail maker objects.
obb_annotator   = sv.OrientedBoxAnnotator(color_lookup=sv.ColorLookup.TRACK, thickness=2)
trace_annotator = sv.TraceAnnotator(trace_length=100, thickness=3, color_lookup=sv.ColorLookup.TRACK)

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    result     = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    #Filtering for only vehicle classes.
    mask       = np.isin(detections.class_id, [9, 10])
    detections = detections[mask]
    detections = tracker.update(detections)
    frame      = obb_annotator.annotate(scene=frame.copy(), detections=detections)
    frame      = trace_annotator.annotate(scene=frame, detections=detections)
    obb_coords = detections.data["xyxyxyxy"]
    #Drawing a centre red dot in each OBB.
    for poly in obb_coords:
        pts = poly.reshape(4, 2).astype(int)
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.circle(frame, (cx, cy), radius=3, color=(0, 0, 255), thickness=-1)
    return frame

sv.process_video(
    #The input file path.
    source_path=r"C:\Users\Xeron\Videos\PrayagIntersection\PrayagIntersection1.mp4",
    #The output file path.
    target_path=r"C:\Users\Xeron\Videos\PrayagIntersection\PrayagIntersection1ProcessedPathTracker.mp4",
    callback=callback
)
