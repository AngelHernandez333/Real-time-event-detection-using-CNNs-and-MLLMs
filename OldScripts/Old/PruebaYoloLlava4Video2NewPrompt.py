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
import numpy as np
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Clases de detección
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
# Contador de las detecciones
classes = dict()
# Inicializadas en cero
for i in detection_labels.values():
    classes[i] = 0


def detection(image):
    """
    Funcion de deteccion de objetos en la imagen
    image: imagen a procesar
    """
    start_time = time.time()
    results = ov_qmodel(image)
    for i in detection_labels.values():
        classes[i] = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
            # Obtener la clase y la confianza
            class_label = int(box.cls[0])  # Convertir a entero si es necesario
            confidence = float(box.conf[0])  # Convertir a flotante si es necesario
            # Muestra la clase y el grado de confianza en el cuadro
            text = f"Class: {detection_labels[class_label] }-{confidence:.2f}"
            classes[detection_labels[class_label]] = (
                classes[detection_labels[class_label]] + 1
            )
            cv2.putText(
                image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
            )
            confidence = math.ceil((box.conf[0] * 100)) / 100
    elapsed_time = time.time() - start_time
    print(elapsed_time * 1000, " ms\n")
    cv2.putText(
        image,
        f"Time {elapsed_time*1000} ms",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        1,
    )


# Elaboracion del prompt para el modelo
def prompt_text(classes):
    initial = "There are"
    objects = ""
    counter = 0
    for x, y in classes.items():
        if y > 0 and counter < 3:
            objects += f" {y} {x},"
            if y > 1:
                objects = objects[:-1] + "s,"
            counter += 1
    objects = objects[:-1]
    text = f"{initial}{objects} in the video,"
    return text


# Validacion de la deteccion
def validation(frames):
    text = prompt_text(classes)
    conversation3 = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {
                    "type": "text",
                    "text": f"{text} there is an possible car accident? Just yes or no",
                },
            ],
        },
    ]
    video = np.stack(frames)
    prompt = processor.apply_chat_template(conversation3, add_generation_prompt=True)
    inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to(
        "cuda:0", torch.float16
    )
    out = model.generate(**inputs, max_new_tokens=60)
    text_outputs = processor.batch_decode(
        out, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    Results[text_outputs[0].split("\n")[-1]] = (
        Results[text_outputs[0].split("\n")[-1]] + 1
    )
    print(text_outputs[0].split("\n")[-1])
    # prompts.append(text_outputs[0].split("\n")[-1])
    prompts.append(text_outputs[0])


# Funcion para tomar los frames y realizar la validacion
def take_frame(frame, frame_number):
    if frame_number % 30 == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) > 5:
            frames.pop(0)
            validation(frames)


start_time = time.time()
ov_qmodel = YOLOv10("/home/ubuntu/yolov10/int8/yolov10x_openvino_model/")
# MLLM Llava
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
model.to(dtype=torch.float16, device="cuda")

# cap = cv2.VideoCapture('../choque_probable_15fps.mp4')
cap = cv2.VideoCapture("../992_aglomeracion_15.mp4")
# cap.set(cv2.CAP_PROP_POS_MSEC, 370000)
cap.set(cv2.CAP_PROP_POS_MSEC, 50000)
prev_frame_time = 0
new_frame_time = 0
frames = []
prompts = []
Results = {"Yes": 0, "No": 0}
elapsed_time = time.time() - start_time

while True:
    # Leer el siguiente frame
    ret, frame = cap.read()
    if not ret:
        print("No se pudo obtener el frame. Fin del video o error.")
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    detection(frame)
    take_frame(frame, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    # -------------------------------------
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("Los FPS son", fps)
cap.release()
cv2.destroyAllWindows()
print("Charging time:", elapsed_time, "sg \n\n")

for i in range(len(prompts)):
    print(prompts[i], "\n")
print(Results, "\n---------------------------------------------\n")
