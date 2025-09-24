import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download
import cv2
from CLIPS import XCLIP_Model


def read_video_opencv(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        if len(frames) == 8:
            break
    return frames

    cap.release()
    return frames  # list of np.ndarrays


def sample_frame_indices(clip_len, frame_sample_rate, total_frames):
    """
    Same logic as original, but works with list of frames.

    Args:
        clip_len (int): Number of frames you want to sample.
        frame_sample_rate (int): Interval between samples (every n-th frame).
        total_frames (int): Total frames available.

    Returns:
        list[int]: Indices to sample.
    """
    converted_len = int(clip_len * frame_sample_rate)
    if total_frames < converted_len:
        raise ValueError("Video is too short for the requested sampling.")

    end_idx = np.random.randint(converted_len, total_frames)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# Supón que ya tienes tus frames:
rute = "/home/ubuntu/Database/ALL/Videos/Riding"
file = "3_105_1.mp4"

sampled_frames = read_video_opencv(f"{rute}/{file}")
CLIP_encoder = XCLIP_Model()
CLIP_encoder.set_model("microsoft/xclip-base-patch32")
CLIP_encoder.set_processor("microsoft/xclip-base-patch32")
descriptions = [
    "a video of a person riding a bicycle",
    "a video of a normal view (persons walking and standing)",
]
CLIP_encoder.set_descriptions(descriptions)
event, avg_prob, logits = CLIP_encoder.outputs_without_softmax(
    sampled_frames[2:], sampled_frames[0:2]
)
logits_np = logits.to(dtype=torch.float32, device="cpu").numpy()[0]
print(event, avg_prob, logits_np)  # Imprime el evento y la probabilidad promedio
"""print(len(sampled_frames), sampled_frames[0].shape)  # Verifica la cantidad de frames y su forma

# Convierte a np.ndarray final
result = np.stack(sampled_frames)  # shape (clip_len, H, W, 3)
device = "cuda"
torch_dtype = torch.float16
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32", torch_dtype=torch_dtype, device_map=device)
import time

start_time= time.time()
inputs = processor(
    text=["riding a bicycle", "normal view (persons walking and standing)"],
    videos=list(result),
    return_tensors="pt",
    padding=True,
).to(device)
with torch.no_grad():
    with torch.autocast(device):
        outputs = model(**inputs)
logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
print(logits_per_video)
probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs, time.time() - start_time)"""
