import cv2
import numpy as np
from SuppotDMC2 import eventsCheck
from DMC_OPP import *

classes_focus = {
    "a person riding a bicycle": ["person", "bicycle"],
    "a certain number of persons fighting": ["person"],
    "a group of persons playing": [
        "person",
        "frisbee",
        "sports ball",
        "baseball glove",
        "tennis racket",
    ],
    "a person running": ["person"],
    "a person lying in the floor": ["person"],
    "a person chasing other person": ["person"],
    "everything is normal": [
        "person",
        "bicycle",
        "frisbee",
        "sports ball",
        "baseball glove",
        "tennis racket",
    ],
    "a person jumping": ["person"],
    "a person falling": ["person"],
    "a person guiding other person": ["person"],
    "a person discarding garbage": ["person"],
}

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
def decision_maker(event):
    match event:
        case "a person riding a bicycle":
            dmc=EventBicycle()
        case 'a certain number of persons fighting':
            dmc=EventFight()
        case 'a group of persons playing':
            dmc=EventPlaying()
        case 'a person running':
            dmc=EventRunning()
        case 'a person lying in the floor':
            dmc=EventLying()
        case 'a person chasing other person':
            dmc=EventChasing()
        case 'a person jumping':
            dmc=EventJumping()
        case 'a person falling':
            dmc=EventFalling()
        case 'a person guiding other person':
            dmc=EventGuiding()
        case 'a person discarding garbage':
            dmc=EventGarbage()
        case "a person stealing other person":
            dmc=EventStealing()
    return dmc

def decision_makerComplex(
    frame,
    frame_number,
    frames,
    gap,
    classes,
    detections,
    results,dcm,
    MLLM=True,
    frames_number=[],
    prompts=[],
):
    if frame_number % gap == 0:
        condition, prompt = dcm.decision_maker(classes, detections,results, frames, MLLM)
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
