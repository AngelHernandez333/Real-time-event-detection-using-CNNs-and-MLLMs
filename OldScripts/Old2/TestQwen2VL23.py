from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import time
from PIL import Image
import requests
from torchvision import io
from typing import Dict
import cv2
import numpy as np
import torch
from math import sqrt


# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
# )
def resize_frame(frame, max_pixels):
    # Calculate aspect ratio and target dimensions based on max_pixels
    h, w, _ = frame.shape
    aspect_ratio = w / h
    target_area = max_pixels
    target_h = int(sqrt(target_area / aspect_ratio))
    target_w = int(aspect_ratio * target_h)
    resized_frame = cv2.resize(
        frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    )
    return resized_frame


# Messages containing a images list as a video and a text query
def validation(frames, model, processor):

    start = time.time()
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                },
                {"type": "text", "text": f"{text} there is {event}? Just yes or no"},
            ],
        }
    ]
    video = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2).float()
    # Preparation for inference
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        videos=video,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text, time.time() - start, "Seconds")


if __name__ == "__main__":
    # Messages containing a video and a text query
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processer
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
    )
    frames = []
    cap = cv2.VideoCapture("jobs.mp4")
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:
            frames.append(frame)
            cv2.imshow("frame", frame)
            cv2.waitKey(100)
        if len(frames) == 6:
            validation(frames, model, processor)
            break
    cap.release()
    print(len(frames), "\n")
