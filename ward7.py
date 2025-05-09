import cv2
video_path = "/home/ubuntu/Database/NWPU_IITB/Videos/Jumping/07.avi"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Width: {width}, Video Height: {height}")

cap.release()