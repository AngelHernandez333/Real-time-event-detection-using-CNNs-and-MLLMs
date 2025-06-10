events = [
        "Riding",
        "Fighting",
        "Playing",
        "Running",
        'Lying',
        "Chasing",
        "Jumping",
        "Falling",
        "Guiding",
        "Stealing",
        "Littering",
        "Tripping",
        "Pickpockering",
    ]

import pandas as pd
import os
import numpy as np
import cv2

rute = "../Database/NWPU_IITB/Videos"
for video_kind in range(len(events)):
        actual_rute = f"{rute}/{events[video_kind]}/"
        files = os.listdir(actual_rute)
        for j in range(len(files)):  # Pasar por todos los videos de la carpeta
            video_path = f"{actual_rute}/{files[j]}"  # Replace with your video file path
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Cannot open video.")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"Video Width: {width}, Video Height: {height}, FPS: {fps}")

            cap.release()