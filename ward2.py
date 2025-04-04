import cv2 
import os

descriptions = [
    "a person riding a bicycle",
    "a certain number of persons fighting",
    "a group of persons playing",
    "a person running",
    "a person lying in the floor",
    "a person chasing other person",
    "a person jumping",
    "a person falling",
    "a person guiding other person",
    "a person stealing other person",
    "a person throwing trash in the floor",
    "a person tripping",
    "a person stealing other person's pocket",
]
rute = f"/home/ubuntu/Tesis/Temp/"
files = os.listdir(rute)
images = []

for i in range(7):
    image_path = f"/home/ubuntu/Tesis/Temp/{files[i]}"

    image=cv2.imread(image_path)

    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Test', frame)
    cv2.waitKey(0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('Test 2', frame)
    cv2.waitKey(0)