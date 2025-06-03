'''
A program to track vehicles in a top‐down aerial view video, using YOLO11m‐seg pretrained model
for segmented masked images detection of vehicles, and DeepSORT with IBN feature extractor for ReID, for tracking the detected vehicles.
This script processes a video, detecting and tracking vehicles while applying segmentation masks and enhanced visualizations.

the VeriWild ReID model is used for feature extraction. The pretrained model was downloaded from:
https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md
'''

import torch
from ultralytics import YOLO
import supervision as sv
from trackers import DeepSORTTracker
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

#The custom IBN feature extractor object for feature extraction (ResNet50‐IBN).
class IBNFeatureExtractor:
    def __init__(self, checkpoint_path: str, device: str):
        self.device = device
        self.model = torch.hub.load(
            'XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=False
        ).to(device).eval()
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model_dict = self.model.state_dict()
        filtered = {}
        for k, v in state.items():
            name = k.replace("backbone.", "")
            if name in model_dict and v.size() == model_dict[name].size():
                filtered[name] = v
        model_dict.update(filtered)
        self.model.load_state_dict(model_dict)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

    def __call__(self, chips: list[np.ndarray]) -> np.ndarray:
        #Handle empty chips list.
        if not chips:
            return np.array([])
        
        #Filter out invalid chips (empty or too small).
        valid_chips = []
        for chip in chips:
            if chip.size > 0 and chip.shape[0] > 10 and chip.shape[1] > 10:
                valid_chips.append(chip)
        
        if not valid_chips:
            return np.array([])
        
        imgs = [self.transform(Image.fromarray(c)) for c in valid_chips]
        batch = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
        return feats.cpu().numpy()

    #The feature extraction method for the DeepSORT tracker.
    #It extracts features from the detected bounding boxes in the frame.
    def extract_features(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        if len(detections) == 0:
            return np.array([])
        
        boxes = detections.xyxy.astype(int)
        chips = []
        for x1, y1, x2, y2 in boxes:
            #Ensure coordinates are within frame bounds.
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            #Check if it's a Valid bounding box.
            if x2 > x1 and y2 > y1:  
                chip = frame[y1:y2, x1:x2]
                if chip.size > 0:
                    chips.append(chip)
        
        return self.__call__(chips)


#Load the pretrained YOLO11m‐seg model for vehicle segmentation in a top‐down aerial view.
model = YOLO(r"C:\Users\Xeron\Videos\PrayagIntersection\yolo11m-seg.pt")

#Check YOLO class names to verify vehicle class indices.
print("YOLO model class names:")
print(model.names)

#Define and initialize the IBN Feature Extractor for ReID.
extractor = IBNFeatureExtractor(
    checkpoint_path=r"C:\Users\Xeron\Videos\PrayagIntersection\veriwild_bot_R50-ibn.pth",
    device="cuda"
)

#The DeepSORT tracker object (uses the IBN extractor internally).
tracker = DeepSORTTracker(
    feature_extractor=extractor,
    frame_rate=50,
    lost_track_buffer=100,
    #Uses the Nvidia GPU for processing.
    device="cuda"
)

#Enhanced visualization components.
mask_annotator = sv.MaskAnnotator(
    color_lookup=sv.ColorLookup.TRACK,
    #Transluscent segmentation masks.
    opacity=0.4
)

#The trajectory trails with fade effect from the center of the bounding boxes.
trace_annotator = sv.TraceAnnotator(
    #The trail length and thickness.
    trace_length=150,
    thickness=4,
    color_lookup=sv.ColorLookup.TRACK,
    position=sv.Position.CENTER
)

#The label annotator with better styling
label_annotator = sv.LabelAnnotator(
    color_lookup=sv.ColorLookup.TRACK,
    text_color=sv.Color.WHITE,
    text_scale=0.8,
    text_thickness=2,
    text_padding=8,
    border_radius=10
)

#The Halo annotator for better text visibility
halo_annotator = sv.HaloAnnotator(
    color_lookup=sv.ColorLookup.TRACK,
    opacity=0.8
)

#The Corner annotator for dynamic visual elements
corner_annotator = sv.BoxCornerAnnotator(
    color_lookup=sv.ColorLookup.TRACK,
    thickness=4,
    corner_length=25
)

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    print(f"Processing frame {index}")
    
    #Run YOLO segmentation inference on the current frame.
    #Filter for only target classes (persons and vehicles).
    #Classes: 0: person, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus, 6: train, 7: truck
    result = model(frame, conf=0.15, iou=0.3, classes=[0, 1, 2, 3, 4, 5, 6, 7])[0]

    #Convert Ultralytics result to Supervision Detections (includes masks, xyxy, class_id).
    detections = sv.Detections.from_ultralytics(result)
    
    print(f"Initial detections: {len(detections)}")
    print(f"Classes detected: {detections.class_id if len(detections) > 0 else 'None'}")

    #Filter strictly for target classes only (persons and vehicles)
    #Class 0: person, Class 1: bicycle, Class 2: car, Class 3: motorcycle, 
    #Class 4: airplane, Class 5: bus, Class 6: train, Class 7: truck
    target_classes = [0, 1, 2, 3, 4, 5, 6, 7]  #Only these specific classes
    if len(detections) > 0:
        #Additional filtering for valid bounding box sizes (min area threshold).
        areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
        #Minimum bounding box area for filtering.
        min_area = 350  
        
        #Combine class and size filtering.
        class_mask = np.isin(detections.class_id, target_classes)
        size_mask = areas >= min_area
        combined_mask = class_mask & size_mask
        
        detections = detections[combined_mask]
    
    print(f"Strictly filtered detections (classes 0-7, min area 400px): {len(detections)}")

    if len(detections) == 0:
        return frame  #Return original frame if no target objects detected.

    #Update the DeepSORT tracker (using bounding boxes under the hood).
    detections = tracker.update(detections, frame)
    
    print(f"Tracked detections: {len(detections)}")

    annotated_frame = frame.copy()
    #Draw segmentation masks first (background layer).
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    #Draw trajectory trails from center.
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

    #Enhanced center points with different colors based on object type.
    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        #Get object class for color coding.
        if i < len(detections.class_id):
            obj_class = detections.class_id[i]
            if obj_class == 0:  #Person
                color = (255, 255, 100)  #Light cyan
            elif obj_class == 1:  #Bicycle
                color = (100, 255, 255)  #Light yellow
            elif obj_class == 2:  #Car
                color = (255, 100, 100)  #Light blue
            elif obj_class == 3:  #Motorcycle
                color = (100, 255, 100)  #Light green
            elif obj_class == 4:  #Airplane
                color = (255, 100, 255)  #Light magenta
            elif obj_class == 5:  #Bus
                color = (100, 100, 255)  #Light red
            elif obj_class == 6:  #Train
                color = (150, 150, 255)  #Light purple
            elif obj_class == 7:  #Truck
                color = (255, 255, 100)  #Light cyan
            else:
                color = (255, 255, 255)  #White for unknown
        else:
            color = (255, 255, 255)
        
        #Draw enhanced center point with glow effect.
        cv2.circle(annotated_frame, (cx, cy), 8, (0, 0, 0), -1)  #Black outline
        cv2.circle(annotated_frame, (cx, cy), 6, color, -1)  #Colored center
        cv2.circle(annotated_frame, (cx, cy), 3, (255, 255, 255), -1)  #White core

    #Add halo effect around labels for better visibility.
    if detections.tracker_id is not None:
        #Create enhanced labels with object type and ID.
        labels = []
        for i, tracker_id in enumerate(detections.tracker_id):
            if i < len(detections.class_id):
                obj_class = detections.class_id[i]
                class_name = model.names.get(obj_class, 'Object')
                labels.append(f"{class_name} #{tracker_id}")
            else:
                labels.append(f"Object #{tracker_id}")

        annotated_frame = halo_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections, 
            labels=labels
        )

    return annotated_frame

#Process the input video, frame by frame, saving the result with mask‐tracking overlay.
try:
    sv.process_video(
        source_path=r"C:\Users\Xeron\Videos\PrayagIntersection\ForeignRoundabout.mp4",
        target_path=r"C:\Users\Xeron\Videos\PrayagIntersection\ForeignRoundaboutProcessed.mp4",
        callback=callback
    )
    print("Video processing completed successfully!")
except Exception as e:
    print(f"Error during video processing: {e}")