import cv2
import time
import os
from MLLMs import *
from Detectors import YOLOv10Detector
import pandas as pd
from Functions3 import (
    decision_maker,
    decision_makerComplex,
    classes_focus,
    detection_labels,
)
import numpy as np
from MLLMs import *
from Tester import VideoTester
import pandas as pd
import numpy as np
from CLIPS import CLIP_Model

if __name__ == "__main__":
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
        'a normal view (persons walking or standing)'
    ]
    # Prepare the tester
    tester = EventTesterCLIP()
    test = 1
    if test == 0:
        llava = LLaVA_OneVision()
        llava.set_model("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
        llava.set_processor("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
        tester.set_dataframe("/home/ubuntu/Tesis/Results/TestingDev.csv")
        tester.set_MLLM(llava)
    elif test == 1:
        janus = JanusPro()
        janus.set_model("deepseek-ai/Janus-Pro-1B")
        janus.set_processor("deepseek-ai/Janus-Pro-1B")
        tester.set_dataframe("/home/ubuntu/Tesis/Results/TestingJanusCLIPPhoto.csv")
        tester.set_MLLM(janus)
    elif test == 2:
        qwen2vl = Qwen2_VL()
        qwen2vl.set_model("Qwen/Qwen2-VL-2B-Instruct")
        qwen2vl.set_processor("Qwen/Qwen2-VL-2B-Instruct")
        tester.set_dataframe("/home/ubuntu/Tesis/Results/TestingIsThereQwen.csv")
        tester.set_MLLM(qwen2vl)
    # tester.set_MLLM(llava)
    tester.show_video(False)
    CLIP_encoder = CLIP_Model()
    CLIP_encoder.set_model("openai/clip-vit-base-patch32")
    CLIP_encoder.set_processor("openai/clip-vit-base-patch32")
    # Add a prefix to each description
    #prefix = "a video of "
    prefix = "a photo of "
    descriptions = [prefix + des for des in description]
    CLIP_encoder.set_descriptions(descriptions)
    tester.set_image_encoder(CLIP_encoder)
    # Start the autotesting
    # tester.autotesting(events, description, [0,1,2,3])
    # tester.simple_autotesting(events, description, [0,1,2,3])
    tester.set_rute("../Database/CHAD DATABASE")
    tester.show_video(False)
    tester.simple_autotestingCLIP(events, description, [0, 1, 2, 3, 4])
    # tester.autotesting(events, description, [0,1,2,3,4])
