#This program uses the RoboFlow's RF-DETR Large Model, and the DeepSORT tracker with the VeriWild R50 ReID (from Fast ReID Model Zoo) for vehicle tracking.

import torch
import cv2
import numpy as np
from trackers import DeepSORTTracker
from rfdetr import RFDETRLarge
from PIL import Image
import torchvision.transforms as T
import os
import random

#Global arrays for dynamic colouring, and tracking of paths.
#track_id -> (B, G, R)
track_colors = {}       # track_id -> (B, G, R)
#track_id -> list of (x, y) centers
track_paths = {}        
#Max number of points in the trail path.
MAX_PATH_LEN = 100       

#The IBN feature extractor for the VeriWild IBN ReID.
class IBNFeatureExtractor:
    def __init__(self, checkpoint_path: str, device: str):
        self.device = device
        self.model = torch.hub.load(
            'XingangPan/IBN-Net',
            'resnet50_ibn_a',
            pretrained=True
        ).to(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        m_dict = self.model.state_dict()
        filtered = {
            k.replace("backbone.", ""): v
            for k, v in state.items()
            if k.replace("backbone.", "") in m_dict
               and v.size() == m_dict[k.replace("backbone.", "")].size()
        }
        m_dict.update(filtered)
        self.model.load_state_dict(m_dict)
        self.feat_dim = 2048
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, chips: list[np.ndarray]) -> np.ndarray:
        if not chips:
            return np.zeros((0, self.feat_dim), dtype=np.float32)
        imgs = [self.transform(Image.fromarray(c)) for c in chips]
        batch = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
        return feats.cpu().numpy()

    def extract_features(self, frame: np.ndarray, detections) -> np.ndarray:
        boxes = detections.xyxy.astype(int)
        chips = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes]
        if not chips:
            return np.zeros((0, self.feat_dim), dtype=np.float32)
        return self.__call__(chips)

#The random colour generator for different tracked objects' bounding boxes (for better visualisations).
def get_color_for(track_id: int):
    if track_id not in track_colors:
        track_colors[track_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return track_colors[track_id]

#The callback function to draw the bounding boxes, and trail paths, frame by frame.
def callback(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = model.predict(Image.fromarray(rgb), threshold=0.4)

    #Filter only for vehicles (bicycles, cars, trucks, and vans)
    vehicle_ids = [1, 2, 3, 5, 7]
    mask = np.isin(detections.class_id, vehicle_ids)
    detections = detections[mask]

    tracked = tracker.update(detections, frame)
    out = frame.copy()

    for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
        #Skipping unmatched detections.
        if track_id < 0:
            continue

        x1, y1, x2, y2 = box.astype(int)
        color = get_color_for(track_id)

        #Draw the bounding box.
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=3)

        #Update the paths array, and draw the trail path.
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        pts = track_paths.setdefault(track_id, [])
        pts.append((cx, cy))
        if len(pts) > MAX_PATH_LEN:
            pts.pop(0)
        if len(pts) > 1:
            cv2.polylines(out, [np.array(pts, dtype=np.int32)], False, color, thickness=3)

    return out

#The main driver function.
if __name__ == "__main__":
    #Use the CUDA enabled GPU, but use CPU as a fallback.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #Load the RF-DETR Large Model, pre-trained on the COCO dataset, with 128M parameters.
    print("Loading RF-DETR Large model...")
    model = RFDETRLarge()

    #Initializing the IBN extractor.
    extractor = IBNFeatureExtractor(
        checkpoint_path=r"C:\Users\Xeron\Videos\PrayagIntersection\veriwild_bot_R50-ibn.pth",
        device=DEVICE
    )
    #Using the DeepSORT tracker with VeriWild ReID.
    tracker = DeepSORTTracker(feature_extractor=extractor)

    #The input, and output file paths.
    input_path  = r"C:\Users\Xeron\Videos\PrayagIntersection\ForeignIntersection.mp4"
    output_path = r"C:\Users\Xeron\Videos\PrayagIntersection\ForeignIntersection_tracked.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

#Processing the video, and saving it.
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = callback(frame)
        writer.write(annotated)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames.")

    cap.release()
    writer.release()
    print(f"Output video saved to: {output_path}")
