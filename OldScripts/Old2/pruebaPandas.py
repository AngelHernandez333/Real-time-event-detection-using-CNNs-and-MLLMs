'''import pickle 

with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

def calculate_shared_area(rect1, rect2):
    """
    Calculate the shared area between two rectangles.

    Parameters:
    rect1 (tuple): A tuple (x1, y1, x2, y2) representing the first rectangle.
    rect2 (tuple): A tuple (x1, y1, x2, y2) representing the second rectangle.

    Returns:
    float: The shared area between the two rectangles.
    """
    x1_1, y1_1, x2_1, y2_1 = rect1[2],rect1[3],rect1[4],rect1[5]
    x1_2, y1_2, x2_2, y2_2 = rect2[2],rect2[3],rect2[4],rect2[5]
    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    # Calculate the width and height of the intersection rectangle
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    area_rect1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    # Calculate the area of the intersection rectangle
    inter_area = inter_width * inter_height
    return inter_area/area_rect1


def person_lying(loaded_data):
    for i in range(len(loaded_data[0])):
        print(loaded_data[0][i], '\n---------------------------------------------------------------------\n')
        contador = 0
        for j in range(1, len(loaded_data)):
            for k in range(len(loaded_data[j])):
                if calculate_shared_area(loaded_data[0][i], loaded_data[j][k]) > 0.5:
                    print(j, '-',k,'-',loaded_data[j][k], calculate_shared_area(loaded_data[0][i], loaded_data[j][k]),'\n')
                    contador += 1
                    break
        if contador > 2:
            print(contador)
            return True
    return False

print(person_lying(loaded_data))

'''

import cv2
import numpy as np

# Create a white image
height, width = 1600, 2000  # You can adjust the dimensions as needed
image = np.ones((height, width, 3), np.uint8) * 255

# Define the rectangles
rect1 = ["person", 0.8695207238197327, 1240, 244, 1315, 299]
rect2 = ["person", 0.8561849594116211, 1303, 251, 1377, 308]

# Draw the rectangles on the image
cv2.rectangle(image, (rect1[2], rect1[3]), (rect1[4], rect1[5]), (0, 255, 0), 2)
cv2.rectangle(image, (rect2[2], rect2[3]), (rect2[4], rect2[5]), (0, 0, 255), 2)

# Display the image
cv2.imshow("Image with Rectangles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
