import sys

sys.path.append("/home/ubuntu/Tesis")  # Add the Tesis directory to the Python path
from Detectors import YOLOv10Detector  # Adjust the import path
import cv2
from Functions3 import (
    detection_labels,
)
import numpy as np

ov_qmodel = YOLOv10Detector()
ov_qmodel.set_model("/home/ubuntu/yolov10/yolov10x.pt")
ov_qmodel.set_labels(detection_labels)
cap = cv2.VideoCapture("/home/ubuntu/Database/ALL/Videos/Stealing/3_080_1.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rute = "/home/ubuntu/Tesis/Results/Tesis/Graphics/Grupales3"
for frame_num in range(0, total_frames, 6):  # Start at 0, step by 6
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    detections, _ = ov_qmodel.detection(frame)
    # ov_qmodel.put_detections(detections, frame)
    person = False
    for detection in detections:
        if detection[0] == "person":
            person = True

    if person:
        cv2.imwrite(f"{rute}/frame_{frame_num:04d}.jpg", frame)
        np.save(f"{rute}/frame_{frame_num:04d}.npy", detections)
    if not ret:
        break
    print(f"Read frame {frame_num} (0-based index)")
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # Save or process the frame here

cap.release()
