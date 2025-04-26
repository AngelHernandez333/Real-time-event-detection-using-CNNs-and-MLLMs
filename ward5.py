import cv2

'''import numpy as np

frame1 = dict(np.load("../Database/NWPU_IITB/GT/gt_Avenue.npz"))  # Convert NpzFile to a dictionary
for key in list(frame1.keys()):
    new_key = key.split('.')[0]  # Split the key by the comma and take the first part
    frame1[new_key] = frame1[key]  # Add the new key-value pair
    del frame1[key]  # Remove the old key
for key in frame1.keys():
    print(key)
for key in frame1.keys():
    print(frame1[key])
np.savez("../Database/NWPU_IITB/GT/gt_newavenue.npz", **frame1)'''

# Path to the video file
video_path = "../Database/NWPU_IITB/Videos/Chasing/000212.avi"
#video_path='/home/ubuntu/Database/CHAD DATABASE/1-Riding a bicycle/1_066_1.mp4'
# Open the video file
video = cv2.VideoCapture(video_path)

# Check if the video file was successfully opened
if not video.isOpened():
    print("Error: Could not open video.")
else:
    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")
    fps=float(round(fps))
    print(fps//6)

# Release the video capture object
video.release()