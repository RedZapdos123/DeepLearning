#The post processor for the intersection videos, for addition of a legend, and better frame rendering, for better visualisation.

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#Define the input and output video file paths.
input_video = r"C:\Users\Xeron\Videos\PrayagIntersection\predict\PrayagIntersection1.avi"
output_video = r"C:\Users\Xeron\Videos\PrayagIntersection\FinalOutput\PrayagIntersection1.avi"

#Open the input video.
cap = cv2.VideoCapture(input_video)
#Error handling
if not cap.isOpened():
    print("Error: Unable to open the video file:", input_video)
    exit(1)

#Fetching the video properties.
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#Creating the VideoWriter object for the output video.
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

#Adjusting the font size and type, for a better legend.
font_path = r"C:\Windows\Fonts\Arial.ttf"
font_size = 36

#Loading the font, with error handling for the font file.
try:
    font = ImageFont.truetype(font_path, font_size)
except Exception as e:
    print("Error: Unable to load the font. Using the default font.", e)
    font = ImageFont.load_default()

#Define legend text and colors.
legend_text1 = "Large vehicle"
legend_text2 = "Small vehicle"
color1 = (128, 0, 128)   # Purple for Large Vehicles class.
color2 = (0, 0, 255)     # Blue for Small Vehicles class.

#Define Legend text position at top left corner.
pos1 = (10, 10)
pos2 = (10, 10 + font_size + 10)

outline_range = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Converting the frame from BGR to RGB for PIL.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    #Draw text with an outline for "Large vehicle".
    for dx in range(-outline_range, outline_range + 1):
        for dy in range(-outline_range, outline_range + 1):
            if dx != 0 or dy != 0:
                draw.text((pos1[0] + dx, pos1[1] + dy), legend_text1, font=font, fill=(0, 0, 0))
    draw.text(pos1, legend_text1, font=font, fill=color1)

    #Draw text with an outline for "Small vehicle".
    for dx in range(-outline_range, outline_range + 1):
        for dy in range(-outline_range, outline_range + 1):
            if dx != 0 or dy != 0:
                draw.text((pos2[0] + dx, pos2[1] + dy), legend_text2, font=font, fill=(0, 0, 0))
    draw.text(pos2, legend_text2, font=font, fill=color2)

    #Convert the PIL image back to a numpy array in BGR format.
    frame_rgb = np.array(pil_img)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    #Write the frame with the legend to the output video.
    out.write(frame_bgr)

    #Display the frame.
    cv2.imshow("Video with Legend", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
