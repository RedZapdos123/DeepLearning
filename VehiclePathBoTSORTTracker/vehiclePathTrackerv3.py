"""This program does vehicles' detection using the yolo11m-obb pretrained model, and does vehicle tracking with YOLO in-built
   Bot-SORT algorithm with the YOLO inbuilt ReID pretrained model (taken from FastReID Model Zoo) to do vehicle tracking with Unique IDs,
   and do their trail or path tracking by plotting the vehicles' centres, with different colours for better distinguishibility, and
   visulisation. Only small and large vehicle classes are detected, and tracked.
"""

from ultralytics import YOLO
import cv2
import os
import numpy as np
import colorsys

#Load the YOLO model.
model = YOLO(r"C:\Users\Xeron\Videos\PrayagIntersection\yolo11m-obb.pt")
#The input video file path.
inputVideo = r"C:\Users\Xeron\Videos\PrayagIntersection\ForeignRoundabout.mp4"

#Get the video properties using OpenCV.
cap = cv2.VideoCapture(inputVideo)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

#The output directory setup.
#The folder where the root directory.
projectPath = r"C:\Users\Xeron\Videos\PrayagIntersection"
#The folder where the processed video wll be stored.
name = "trackPath"
outputDir = os.path.join(projectPath, name)
os.makedirs(outputDir, exist_ok=True)
#The output video file path, and name.
outputVideoPath = os.path.join(outputDir, "ForeignRoundaboutVehiclePathTracked.mp4")

#Initialize video writer.
#Using mp4 video format for processing.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(outputVideoPath, fourcc, fps, (width, height))

#The Function to Get Bright, Distinguishable Colors for a Track ID, for better visualisation.
def get_color_for_track(trackID):
    #Ensure that the trackID is an integer.
    trackID = int(trackID)
    #Multiply by the golden ratio conjugate and take modulo 1 for hue.
    hue = (trackID * 0.618033988749895) % 1.0
    #Convert from HSV (with full saturation and brightness) to RGB.
    r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
    #Return values scaled to 255.
    return (int(r * 255), int(g * 255), int(b * 255))


#The permanent botsort.yaml tracker configuration file (using the BoT-SORT algoritim built into YOLO).
trackerYAMLpath = r"C:\Users\Xeron\OneDrive\Documents\Programs\MachineLearning\IIITAFacesDataset\PrayagIntersectionPrograms\botsort.yaml"


#The Tracking history dictionary => keys: track IDs; values: list of (x, y) coordinates
trackingHistory = {}
#The missed frames counter => keys: track IDs; values: number of consecutive frames missed
missedFrames = {}
#Maximum allowed missing frames before a track is removed.
MISS_THRESHOLD = 100

#Run the tracking process using the streaming interface.
results = model.track(
    inputVideo,
    show=False,
    show_labels=False, #Removed the labels for better visualisation.
    show_conf=False, #Removed confidence score labels for better visulisation
    line_width=2,
    classes=[9, 10],  #Filter for only vehicle classes.
    tracker=trackerYAMLpath, #The .yaml tracker file path (using botsort algorithm with VERI-WILD ReID)
    save=False,
    stream=True,
    conf=0.15 #Decreased the confidence score for increased number of detections, and trackings.
)

#Processing each frame from the stream.
for result in results:
    #Get the plotted image with overlays.
    plottedImg = result.plot(labels=False, conf=False, line_width=2)
    currentRedCenters = []
    currentTrackIDs = set()

    #Processing the oriented bounding boxes (OBB) from the model.
    if hasattr(result, 'obb') and result.obb is not None:
        for idx, box in enumerate(result.obb):
            #Extract center coordinates from the OBB's xywhr information.
            cx = int(box.xywhr[0][0].item())
            cy = int(box.xywhr[0][1].item())
            currentRedCenters.append((cx, cy))

            #Retrieve the unique tracking ID if available, otherwise use the index.
            trackID = getattr(box, 'id', None)
            if trackID is None:
                trackID = idx
            currentTrackIDs.add(trackID)

            #Update tracking history with the new center coordinate.
            if trackID not in trackingHistory:
                trackingHistory[trackID] = []
            trackingHistory[trackID].append((cx, cy))
            #Limit each track's history to the most recent 400 points.
            if len(trackingHistory[trackID]) > 400:
                trackingHistory[trackID] = trackingHistory[trackID][-400:]

            #Reset missed frame counter for this track.
            missedFrames[trackID] = 0

    #Increment missed frame counters for tracks not detected in the current frame.
    for trackID in list(trackingHistory.keys()):
        if trackID not in currentTrackIDs:
            missedFrames[trackID] = missedFrames.get(trackID, 0) + 1
            if missedFrames[trackID] > MISS_THRESHOLD:
                del trackingHistory[trackID]
                del missedFrames[trackID]

    #Draw trailing paths for each track using a unique bright color determined by its trackID.
    for trackID, pts in trackingHistory.items():
        color = get_color_for_track(trackID)  # Get unique bright color for this track
        if len(pts) >= 2:
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(plottedImg, [pts_np], isClosed=False, color=color, thickness=2)
        elif len(pts) == 1:
            cv2.circle(plottedImg, pts[0], radius=3, color=color, thickness=-1)

    #Draw red dots for current frame's detected center points.
    for (cx, cy) in currentRedCenters:
        cv2.circle(plottedImg, (cx, cy), radius=3, color=(0, 0, 255), thickness=-1)

    #Write the modified frame to the output video.
    out.write(plottedImg)

#Release the video writer when the processing is complete.
out.release()
print(f"The output video with distinguishable and consistent track paths/trails, and vehicles' detections, saved to: {outputVideoPath}")
