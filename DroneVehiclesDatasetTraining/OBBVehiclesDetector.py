from ultralytics import YOLO

#Loading the pretrained YOLO11m-obb model.
model = YOLO(r"C:/Users/Xeron/OneDrive/Documents/Programs/DroneVehiclesDatasetTraining/DroneVehiclesYOLO_OBB_TrainingOutput/weights/DroneVehiclesYOLO11m-OBB.pt")

#Detecting the vehicles in the video.
results = model.predict(
    r"C:\Users\Xeron\Videos\PrayagIntersection\PrayagIntersection1.mp4", #the input video file path.
    show=False,
    #Remove the large labels for better visualisation.
    show_labels=False,
    show_conf=False,
    line_width=2,
    #Save to the project folder for further use.
    save=True,
    project=r"C:\Users\Xeron\Videos\NewOutput",
    name="Predictions",
    #Increased the confidence score limit for lower number of false detections.
    conf= 0.3
)

