import pandas as pd
import os
import numpy as np

events = [
    "1-Riding a bicycle",
    "2-Fight",
    "3-Playing",
    "4-Running away",
    "5-Person lying in the floor",
    "6-Chasing",
    "7-Jumping",
    "8-Falling",
    "9-guide",
    "10-thief",
    "11-Littering",
    "12-Tripping",
    "13-Pickpockering",
]
description = [
        "Riding",
        "Fighting",
        "Playing",
        "Running",
        "Lying",
        "Chasing",
        "Jumping",
        "Falling",
        "Guiding",
        "Stealing",
        "Littering",
        "Tripping",
        "Pickpockering",
    ]

anomaly_frames=0.0
total_frames=0.0
rute = "../Database/CHAD DATABASE/"
for video_kind in range(len(events)):
        actual_rute = f"{rute}/{events[video_kind]}/"
        files = os.listdir(actual_rute)
        for j in range(len(files)):  # Pasar por todos los videos de la carpeta
            name = files[j].split(".")[0]
            frames = np.load(
                f"../Database/CHAD DATABASE/CHAD_Meta/anomaly_labels/{name}.npy"
            )
            anomaly_frames += frames.sum()
            total_frames += frames.shape[0]

print(anomaly_frames, total_frames-anomaly_frames ,total_frames)

events = [
        "Riding",
        "Fighting",
        "Playing",
        "Running",
        "Chasing",
        "Jumping",
        "Guiding",
        "Stealing",
        "Littering",
        "Pickpockering",
    ]
rute = "../Database/NWPU_IITB/Videos"
for video_kind in range(len(events)):
        actual_rute = f"{rute}/{events[video_kind]}/"
        files = os.listdir(actual_rute)
        for j in range(len(files)):  # Pasar por todos los videos de la carpeta
            
            name = files[j].split(".")[0]
            frames = np.load("../Database/NWPU_IITB/GT/gt.npz")
            frames= frames[name]
            frames = np.append(frames, frames[-1])
            print(frames.sum(),  frames.shape[0])
            anomaly_frames += frames.sum()
            total_frames += frames.shape[0]

print(anomaly_frames, total_frames-anomaly_frames ,total_frames)