import cv2
import numpy as np
import sys

sys.path.append("/home/ubuntu/Tesis")  # Add the Tesis directory to the Python path
from DMC_OPP import *

# detections= np.load('/home/ubuntu/Tesis/Results/Tesis/Graphics/OrganizePersons/frame_0102.jpg.npy', allow_pickle=True)

step = 6
ending = 6 * 89
starting = ending - (6 * 6)
rute = "/home/ubuntu/Tesis/Results/Tesis/Graphics/Grupales3"
files = [f"frame_{starting:04d}.npy" for starting in range(starting, ending, step)]
file_name = files[-1].split(".")[0]
image = cv2.imread(f"{rute}/{file_name}.jpg")
file_name2 = files[0].split(".")[0]
image_last = cv2.imread(f"{rute}/{file_name2}.jpg")
cv2.imwrite(f"{rute}/Stored/{file_name}.jpg", image)
cv2.imwrite(f"{rute}/Stored/{file_name2}.jpg", image_last)
all = []
for file in files:
    detections = np.load(f"{rute}/{file}", allow_pickle=True).tolist()
    persons = []
    for i in detections:
        if i[0] == "person" and float(i[1]) > 0.7:
            i[2] = int(i[2])
            i[3] = int(i[3])
            i[4] = int(i[4])
            i[5] = int(i[5])
            persons.append(i)
    all.append(persons)
persons = DecisionMakerPerEvent.organize_persons(all)
person_index = DecisionMakerPerEvent.verify_running(persons)
print("\n\n\n\n", len(persons), person_index)
for index in person_index:
    for i in range(len(persons[index])):
        cv2.rectangle(
            image,
            (persons[index][i][2], persons[index][i][3]),
            (persons[index][i][4], persons[index][i][5]),
            (0, ((255 * i) // len(persons[index])), 0),
            3,
        )
index = 1
i = -1
cv2.rectangle(
    image,
    (persons[index][i][2], persons[index][i][3]),
    (persons[index][i][4], persons[index][i][5]),
    (0, ((255 * i) // len(persons[index])), 0),
    3,
)


dmc = EventStealing()
status, person_index = dmc.process_detections(all)
print(person_index)
print(len(person_index), status)

"""for index in person_index:
    for i in range(len(index)):
        cv2.rectangle(image, (index[2], index[3]), (index[4], index[5]), (0, 255, 0), 3)
"""
# person_index = DecisionMakerPerEvent.verify_falling(persons, False)

"""for person in persons:
    for i in range(len(person)):
        cv2.rectangle(image, (person[i][2], person[i][3]), (person[i][4], person[i][5]), (0, ((255*i)//len(persons)), 0), 5)
""" """for person in persons:
    for i in range(1,len(person)):
        prev_center = ((person[i-1][2] + person[i-1][4]) // 2, (person[i-1][3] + person[i-1][5]) // 2)
        curr_center = ((person[i][2] + person[i][4]) // 2, (person[i][3] + person[i][5]) // 2)
        cv2.line(image, prev_center, curr_center, (0, 0, 0), 5)"""
"""for index in person_index:
    for i in range(len(persons[index])):
        cv2.rectangle(image, (persons[index][i][2], persons[index][i][3]), (persons[index][i][4], persons[index][i][5]), (0, ((255*i)//len(persons[index])), 0), 3)
"""
"""for person in person_index:
    for i in range(len(person)):
        cv2.rectangle(image, (person[i][2], person[i][3]), (person[i][4], person[i][5]), (0, ((255*i)//len(persons)), 0), 5)
"""
"""i=-1
prev_center = ((person_index[1][i][2] + person_index[1][i][4]) // 2, (person_index[1][i][3] + person_index[1][i][5]) // 2)
curr_center = ((person_index[0][i][2] + person_index[0][i][4]) // 2, (person_index[0][i][3] + person_index[0][i][5]) // 2)
cv2.line(image, prev_center, curr_center, (0, 255, 0), 5)
cv2.imwrite(f'{rute}/Stored/{file_name}organizepersons.jpg', image)
i=1
prev_center2 = ((person_index[1][i][2] + person_index[1][i][4]) // 2, (person_index[1][i][3] + person_index[1][i][5]) // 2)
curr_center2 = ((person_index[0][i][2] + person_index[0][i][4]) // 2, (person_index[0][i][3] + person_index[0][i][5]) // 2)
cv2.line(image_last, prev_center2, curr_center2, (0, 255, 0), 5)
cv2.imwrite(f'{rute}/Stored/{file_name2}organizepersons.jpg', image_last)
)"""
cv2.imwrite(f"{rute}/Stored/{file_name}organizepersons.jpg", image)
cv2.imwrite(f"{rute}/Stored/{file_name2}organizepersons.jpg", image_last)
cv2.imshow("Image", image)

cv2.waitKey(0)
