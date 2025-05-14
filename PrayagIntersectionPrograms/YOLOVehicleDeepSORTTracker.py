#This program uses YOLO11m-OBB pretrained model for vehicles detection in top-down aerial view,
#Deep SORT Tracker for vehicles' path tracking, the VeriWild R50 ReID for improving the path tracking.
#The OBBs, OBB centres, and trail paths are drawn for better visualization.
import torch
from ultralytics import YOLO
import supervision as sv
from trackers import DeepSORTTracker
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

#The custom ibn feature extractor object for feature extraction ResNet50.
class IBNFeatureExtractor:
    def __init__(self, checkpoint_path: str, device: str):
        self.device = device
        self.model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=False).to(device).eval()
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
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, chips: list[np.ndarray]) -> np.ndarray:
        imgs = [self.transform(Image.fromarray(c)) for c in chips]
        batch = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
        return feats.cpu().numpy()

    def extract_features(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        boxes = detections.xyxy.astype(int)
        chips = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes]
        return self.__call__(chips)

#Loading the pretrained yolo1m-obb model for vehicle detection from a top-down aerial view.
model = YOLO(r"C:\Users\Xeron\Videos\PrayagIntersection\yolo11m-obb.pt")
#Defining, and initializing the IBN Feature Extractor object.
extractor = IBNFeatureExtractor(
    checkpoint_path=r"C:\Users\Xeron\Videos\PrayagIntersection\veriwild_bot_R50-ibn.pth",
    #Using the Nvidia GPU.
    device="cuda"
)
#The DeepSORT tracker object.
tracker = DeepSORTTracker(feature_extractor=extractor)
#The OBB annotator object.
obb_annotator   = sv.OrientedBoxAnnotator(color_lookup=sv.ColorLookup.TRACK, thickness=2)
#The tracer object to draw the path trails.
trace_annotator = sv.TraceAnnotator(trace_length=100, thickness=3, color_lookup=sv.ColorLookup.TRACK)

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    result     = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    mask       = np.isin(detections.class_id, [9, 10])
    detections = detections[mask]
    detections = tracker.update(detections, frame)
    frame      = obb_annotator.annotate(scene=frame.copy(), detections=detections)
    frame      = trace_annotator.annotate(scene=frame, detections=detections)
    #Drawing the centre red dots of the OBBs.
    for poly in detections.data["xyxyxyxy"]:
        pts = poly.reshape(4, 2).astype(int)
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
    return frame

sv.process_video(
    #The input file path.
    source_path=r"C:\Users\Xeron\Videos\PrayagIntersection\PrayagIntersection1.mp4",
    #The output file path.
    target_path=r"C:\Users\Xeron\Videos\PrayagIntersection\PrayagIntersection1ProcessedPathTracker.mp4",
    callback=callback
)
