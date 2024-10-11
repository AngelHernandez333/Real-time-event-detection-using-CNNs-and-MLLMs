import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Load the DataFrame from the CSV file
'''df = pd.read_csv('results.csv')

print(df)
'''
detections = [True, False]
#np.save(f'{number}_{detector_status}.npy', fps_list)
number=0
for number in range(5):
    plt.figure(number)
    for detector_status in detections:
        fps_list = np.load(f'{number}_{detector_status}.npy')
        if detector_status:
            detector = 'ON'
        else:
            detector = 'OFF'
        plt.plot(fps_list[1::], label=f'Detector {detector}')
    plt.title(f'Video ID:{number}')
    plt.xlabel('Frame number')
    plt.ylabel('FPS')
    plt.legend()
    plt.grid()
plt.show()