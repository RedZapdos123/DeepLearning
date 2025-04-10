import cv2
import ffmpeg
import os

#Procesing the video to convert it to .mp4 format with 50FPS frames rate and uniform frame size (landscape 1080p), for ease of processing.
def process_video(input_path, fps=50, width=1920, height=1080):
    
    #Extracting the folder and filename from input path.
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    
    #Creating the output filename.
    name, ext = os.path.splitext(input_filename)
    output_filename = f"{name}_processed.mp4"
    
    #Define output path in the same folder.
    output_path = os.path.join(input_dir, output_filename)

    #Open the video file.
    cap = cv2.VideoCapture(input_path)
    
    #Error handling if the file is not found.
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    #Fetching the original video FPS for checking.
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")

    #Define the (mp4) codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    temp_output = os.path.join(input_dir, "temp_output.mp4")

    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Resize the frames to 1080p, for uniformity.
        frame = cv2.resize(frame, (width, height))
        
        #Write the frame.
        out.write(frame)

    cap.release()
    out.release()

    #Convert to .mp4 format using FFmpeg library.
    ffmpeg.input(temp_output).output(output_path, vcodec='libx264', crf=23).run(overwrite_output=True)

    os.remove(temp_output)

    print(f"Processed video saved as: {output_path}")

#Define the input path.
input_video_path = r"C:\Users\Xeron\Videos\PrayagIntersection\FinalOutput\ForeignIntersection.avi"
process_video(input_video_path)
