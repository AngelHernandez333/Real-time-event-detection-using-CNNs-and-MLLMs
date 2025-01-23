#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:35:39 2024

@author: ubuntu
"""
import cv2
from ultralytics import YOLOv10
import math
import time
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


"""
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

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
def run_example(task_prompt, text_input=None):
    start_time = time.time() 
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    elapsed_time = time.time() - start_time
    print(parsed_answer, " Tiempo de procesmiento", elapsed_time,"(sg)")   
    return parsed_answer
#Cargar el modelo de Yolov10-x
ov_qmodel=YOLOv10('/home/ubuntu/yolov10/int8/yolov10x_openvino_model/') 
#Establecer el tiempo y la imagen a usar
start_time = time.time()
img=cv2.imread("/home/ubuntu/Descargas/car.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convertir el array NumPy a una imagen PIL para el MLLM
image = Image.fromarray(img_rgb)
#Realizar la deteccion con el model
results=ov_qmodel(img)
#DIbujar las detecciones 
for r in results:
    boxes= r.boxes
    for box in boxes:
        x1,y1,x2,y2=box.xyxy[0]
        x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
        # Obtener la clase y la confianza
        class_label = int(box.cls[0])  # Convertir a entero si es necesario
        confidence = float(box.conf[0])  # Convertir a flotante si es necesario
        # Muestra la clase y el grado de confianza en el cuadro
        text = f"Class: {detection_labels[class_label] }, Confidence: {confidence:.2f}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        confidence=math.ceil((box.conf[0]*100))/100
        print("Confidence -->", confidence)
elapsed_time = time.time() - start_time
timetext=f"Time {elapsed_time*1000} ms" 
print(elapsed_time,"\n\n")
cv2.putText(img, f"Time {elapsed_time*1000} ms" , (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
#Mostrar la imagen de las detecciones
cv2.imshow("Detections", img)
cv2.waitKey(3000)
#Validacion con MLLM
#Prompt
task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
#Llamar a la funcion correspondiente
results = run_example(task_prompt, text_input="A green car parked in front of a yellow building.")
#Observar las detecciones del MLLM
img=cv2.imread("/home/ubuntu/Descargas/car.jpg")
for i in range(len(results['<CAPTION_TO_PHRASE_GROUNDING>']["bboxes"])):
    x1,y1,x2,y2=results['<CAPTION_TO_PHRASE_GROUNDING>']["bboxes"][i]
    x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
    # Obtener la clase y la confianza
    # Muestra la clase y el grado de confianza en el cuadro
    label =results['<CAPTION_TO_PHRASE_GROUNDING>']["labels"][i]
    text = f"Class: {label }"
    cv2.putText(img, text, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
print("Finished.")
cv2.imshow("MLLM", img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
