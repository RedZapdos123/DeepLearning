#This program tracks vehicle paths in a video using YOLOv8 and SAHI.
#It uses YOLOv11-OBB for object detection and SAHI for slicing predictions.
#It draws bounding boxes and paths for detected vehicles, maintaining a history of their movements.
#The output is saved as a video file with the tracked paths and detections.

from ultralytics import YOLO
import cv2
import os
import numpy as np
import colorsys
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

#The configuration for YOLOv11-OBB model and input video.
model = YOLO(r"C:\Users\Xeron\Videos\PrayagIntersection\yolo11m-obb.pt")
inputVideo = r"C:\Users\Xeron\Videos\PrayagIntersection\PrayagIntersection1.mp4"

#Initialize video capture and output settings.
cap = cv2.VideoCapture(inputVideo)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#Create output directory for the tracked video.
projectPath = r"C:\Users\Xeron\Videos\PrayagIntersection"
name = "trackPath"
outputDir = os.path.join(projectPath, name)
os.makedirs(outputDir, exist_ok=True)
#Define the output video path.
outputVideoPath = os.path.join(outputDir, "PrayagIntersection1VehiclePathTrackedSAHI.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(outputVideoPath, fourcc, fps, (width, height))

#Initialize the SAHI detection model with YOLOv11-OBB.
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=r"C:\Users\Xeron\Videos\PrayagIntersection\yolo11m-obb.pt", #Path to the YOLOv11-OBB model.
    confidence_threshold=0.1,
    device="0" #The device to run the model on.
)

#Function to get a color for each track ID based on its unique identifier.
def get_color_for_track(trackID):
    trackID = int(trackID)
    hue = (trackID * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    return (int(r * 255), int(g * 255), int(b * 255))

#Function to draw oriented bounding boxes (OBB) on the image.
def draw_obb(img, box, color, thickness=2):
    if hasattr(box, 'xyxyxyxy'):
        points = box.xyxyxyxy[0].cpu().numpy().astype(int)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(img, [points], True, color, thickness)
    elif hasattr(box, 'xywhr'):
        cx, cy, w, h, r = box.xywhr[0].cpu().numpy()
        rect = ((cx, cy), (w, h), np.degrees(r))
        box_points = cv2.boxPoints(rect)
        box_points = np.int32(box_points)
        cv2.drawContours(img, [box_points], 0, color, thickness)

#Function to draw bounding boxes on the image.
def draw_bbox(img, box, color, thickness=2):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

#The path to the tracker YAML configuration file.
trackerYAMLpath = r"C:\Users\Xeron\OneDrive\Documents\Programs\SahiVehiclePathTracker\botsort.yaml"

trackingHistory = {}
missedFrames = {}
MISS_THRESHOLD = 100
frame_count = 0

#Start processing the video frame by frame.
print(f"Processing video: {inputVideo}")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    
    detections = []
    for object_prediction in result.object_prediction_list:
        if object_prediction.category.id in [9, 10]:
            bbox = object_prediction.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            conf = object_prediction.score.value
            cls = object_prediction.category.id
            detections.append([x1, y1, x2, y2, conf, cls])
    
    print(f"Frame {frame_count}: SAHI detections = {len(detections)}")
    
    if detections:
        detections_array = np.array(detections)
        
        results = model.track(
            source=frame,
            persist=True,
            tracker=trackerYAMLpath,
            classes=[9, 10],
            conf=0.05,
            verbose=False,
            device="0"
        )
        
        if results and len(results) > 0 and results[0] is not None:
            result = results[0]
            plottedImg = frame.copy()
            
            if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                print(f"Frame {frame_count}: OBB detections = {len(result.obb)}")
                currentRedCenters = []
                currentTrackIDs = set()
                
                for idx, box in enumerate(result.obb):
                    cx = int(box.xywhr[0][0].item())
                    cy = int(box.xywhr[0][1].item())
                    currentRedCenters.append((cx, cy))
                    
                    trackID = getattr(box, 'id', None)
                    if trackID is not None:
                        trackID = int(trackID.item())
                    else:
                        trackID = f"obb_{idx}"
                    currentTrackIDs.add(trackID)
                    
                    if trackID not in trackingHistory:
                        trackingHistory[trackID] = []
                    trackingHistory[trackID].append((cx, cy))
                    if len(trackingHistory[trackID]) > 100:
                        trackingHistory[trackID] = trackingHistory[trackID][-100:]
                    
                    missedFrames[trackID] = 0
                    
                    color = get_color_for_track(trackID)
                    draw_obb(plottedImg, box, color)
                    
            elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                print(f"Frame {frame_count}: Box detections = {len(result.boxes)}")
                currentRedCenters = []
                currentTrackIDs = set()
                
                for idx, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    currentRedCenters.append((cx, cy))
                    
                    trackID = getattr(box, 'id', None)
                    if trackID is not None:
                        trackID = int(trackID.item())
                    else:
                        trackID = f"box_{idx}"
                    currentTrackIDs.add(trackID)
                    
                    if trackID not in trackingHistory:
                        trackingHistory[trackID] = []
                    trackingHistory[trackID].append((cx, cy))
                    if len(trackingHistory[trackID]) > 100:
                        trackingHistory[trackID] = trackingHistory[trackID][-100:]
                    
                    missedFrames[trackID] = 0
                    
                    color = get_color_for_track(trackID)
                    draw_bbox(plottedImg, box, color)
            else:
                print(f"Frame {frame_count}: No valid detections from YOLO")
                currentRedCenters = []
                currentTrackIDs = set()
        else:
            print(f"Frame {frame_count}: No YOLO results")
            plottedImg = frame.copy()
            currentRedCenters = []
            currentTrackIDs = set()
    else:
        print(f"Frame {frame_count}: No SAHI detections, running YOLO directly")
        results = model.track(
            source=frame,
            persist=True,
            tracker=trackerYAMLpath,
            classes=[9, 10],
            conf=0.05,
            verbose=False,
            device="0"
        )
        
        if results and len(results) > 0 and results[0] is not None:
            result = results[0]
            plottedImg = frame.copy()
            
            if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                for idx, box in enumerate(result.obb):
                    trackID = getattr(box, 'id', None)
                    if trackID is not None:
                        trackID = int(trackID.item())
                    else:
                        trackID = f"obb_{idx}"
                    color = get_color_for_track(trackID)
                    draw_obb(plottedImg, box, color)
                    
            elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                for idx, box in enumerate(result.boxes):
                    trackID = getattr(box, 'id', None)
                    if trackID is not None:
                        trackID = int(trackID.item())
                    else:
                        trackID = f"box_{idx}"
                    color = get_color_for_track(trackID)
                    draw_bbox(plottedImg, box, color)
        else:
            plottedImg = frame.copy()
        currentRedCenters = []
        currentTrackIDs = set()
    
    for trackID in list(trackingHistory.keys()):
        if trackID not in currentTrackIDs:
            missedFrames[trackID] = missedFrames.get(trackID, 0) + 1
            if missedFrames[trackID] > MISS_THRESHOLD:
                del trackingHistory[trackID]
                del missedFrames[trackID]
    
    for trackID, pts in trackingHistory.items():
        color = get_color_for_track(trackID)
        if len(pts) >= 2:
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(plottedImg, [pts_np], isClosed=False, color=color, thickness=3)
        elif len(pts) == 1:
            cv2.circle(plottedImg, pts[0], radius=5, color=color, thickness=-1)
    
    for (cx, cy) in currentRedCenters:
        cv2.circle(plottedImg, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
    
    out.write(plottedImg)

cap.release()
out.release()
print(f"The output video with distinguishable and consistent track paths/trails, and vehicles' detections, saved to: {outputVideoPath}")