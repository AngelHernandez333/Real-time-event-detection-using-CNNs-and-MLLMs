"""
Functions used in the main script
"""

import cv2
import numpy as np
from SuppotDMC2 import eventsCheck, classes_focus

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


# Elaboracion del prompt para el modelo
def prompt_text(classes, event, detector_usage):
    if detector_usage > 2:
        return "Watch the video,"
    initial = "There are"
    objects = ""
    classes_focus[event]
    corrects = []
    for x, y in classes.items():
        if y > 0 and x in classes_focus[event]:
            corrects.append(x)
    if len(corrects) == 1:
        if classes[corrects[0]] == 1:
            objects += f"There is {classes[corrects[0]]} {corrects[0]} in the video"
        else:
            objects += f"There are {classes[corrects[0]]} {corrects[0]}s in the video"
        return objects
    elif len(corrects) > 1:
        for x in corrects:
            if x == corrects[-1]:
                objects += f" and"
            objects += f" {classes[x]} {x},"
            if classes[x] > 1:
                objects = objects[:-1] + "s,"
    if objects == "":
        text = "Watch the video,"
    else:
        objects = objects[:-1]
        text = f"{initial}{objects} in the video,"
    return text


# Funcion para tomar los frames y realizar la validacion
def take_frame(frame, frame_number, frames, gap, detections, results):
    if frame_number % gap == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        results.append(detections)


def check_precision(prompts, frames_number, video_name):
    # True positive Prediction and Reality are true
    # True negative Prediction and Reality are false
    # False negative Prediction is false and Reality is true
    # False positive Prediction is true and Reality is false
    tp = 0
    fp = 0
    fn = 0
    prompts = [prompt.lower() for prompt in prompts]
    name = video_name.split(".")[0]
    frames = np.load(f"../Database/CHAD DATABASE/CHAD_Meta/anomaly_labels/{name}.npy")
    for i in range(len(prompts)):
        print(prompts[i], frames[frames_number[i] - 1], frames_number[i])
        if prompts[i] == "yes" and frames[frames_number[i] - 1] == 1:
            tp += 1
        elif prompts[i] == "yes" and frames[frames_number[i] - 1] == 0:
            fp += 1
        elif prompts[i] == "no" and frames[frames_number[i] - 1] == 1:
            fn += 1
    print(tp, fp, fn)
    try:
        return tp / (tp + fp), tp / (tp + fn)
    except:
        return 0, 0


def decision_makerComplex(
    frame,
    frame_number,
    frames,
    gap,
    classes,
    detections,
    event,
    results,
    MLLM=True,
    frames_number=[],
    prompts=[],
    file="",
):
    if frame_number % gap == 0:
        condition, prompt = eventsCheck(
            event, classes, detections, results, frames, MLLM, frame_number, file
        )
        if condition and MLLM:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            results.append(detections)
        if MLLM == False:
            if prompt == "":
                pass
            else:
                frames_number.append(frame_number)
                prompts.append(prompt)
            results.append(detections)
            if len(results) > 6:
                results.pop(0)
