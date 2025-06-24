import cv2
import os

from scripts.error import PipelineError

## Function to extract images from user uploaded Video.
def extract_frames(video_file, output_dir, frame_rate = 2): 
    cap = cv2.VideoCapture(video_file)
    
    frame_count = 0
    
    # Create an output folder with a name corresponding to the video
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Only extract frames at the desired frame rate
        if frame_count % int(cap.get(5) / frame_rate) == 0:
            output_file = f"{output_dir}/IMG_{frame_count}.jpg"
            cv2.imwrite(output_file, frame)
            print(f"Frame {frame_count} has been extracted and saved as {output_file}")
    
    cap.release()
    cv2.destroyAllWindows()

## Function to find uploaded Video.
def find_video_in_folder(folder_path):
    if not os.path.exists(folder_path):
        raise PipelineError("No such video path exists")
    
    # Supported video file extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(video_extensions) and os.path.isfile(file_path):
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                cap.release()
                return filename
            cap.release()
    
    return None  # No 

def count_colmap_images_recursive(folder_path):
    colmap_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.pgm', '.ppm', '.tif', '.tiff'}
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in colmap_extensions:
                count += 1
    return count
