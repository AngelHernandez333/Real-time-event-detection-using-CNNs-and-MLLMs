
import cv2
from ultralytics import YOLOv10
import math
import time

'''cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4, 480)'''

detection_labels = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}
import torch
import numpy as np
'''
model = YOLOv10('yolov10x.pt') # load an official model
model.export(format="openvino")
ov_model=YOLOv10('yolov10x_openvino_model/') 
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ov_qmodel = YOLOv10('/home/ubuntu/yolov10/int8/yolov10x_openvino_model/') 
#ov_model=YOLOv10('/home/ubuntu/yolov10/yolov10x_openvino_model/') 
model = YOLOv10('/home/ubuntu/yolov10/yolov10x.pt')
i=0
time_values = np.empty((0, 1))
cap = cv2.VideoCapture("../Database/10.mp4")
# Get the width and height of the frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Initialize the VideoWriter object for MP4 format
out = cv2.VideoWriter('output_frames.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))
while True:
    i+=1
    start_time = time.time()
    success, img=cap.read()
    results=model(img, stream=True)
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 20 == 0:
            cv2.imshow("webcam", img)
            if cv2.waitKey(1)==ord('q') or i==120:
                break
            for i in range(1, 20):
                out.write(img)
    #results=ov_model(img)
    #results = ov_qmodel.predict(source=img, device='GPU')
    for r in results:
        boxes= r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1,y1), (x2,y2), (172, 182, 77),2)
            # Obtener la clase y la confianza
            class_label = int(box.cls[0])  # Convertir a entero si es necesario
            confidence = float(box.conf[0])  # Convertir a flotante si es necesario
            
            # Muestra la clase y el grado de confianza en el cuadro
            text = f"Class: {detection_labels[class_label] }, Confidence: {confidence:.2f}"
            #cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (172, 182, 77), 1)
            confidence=math.ceil((box.conf[0]*100))/100
            print("Confidence -->", confidence)
    elapsed_time = (time.time() - start_time)*1000
    #cv2.putText(img, f"Time {elapsed_time:.2f} ms" , (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (172, 182, 77), 2)
cap.release()
out.release()
cv2.destroyAllWindows()