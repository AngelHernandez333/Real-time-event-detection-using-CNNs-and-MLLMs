import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download
import cv2

def read_video_opencv(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

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
rute='/home/ubuntu/Database/ALL/Videos/Riding'
file='3_105_1.mp4'
frames = read_video_opencv(f"{rute}/{file}")
indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, total_frames=len(frames))

# Selecciona los frames requeridos
sampled_frames = [frames[i] for i in indices]

# Convierte a np.ndarray final
result = np.stack(sampled_frames)  # shape (clip_len, H, W, 3)

processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
inputs = processor(
    text=["playing sports", "eating spaghetti", "go shopping"],
    videos=list(frames),
    return_tensors="pt",
    padding=True,
)
msg_token = msg_token.view(batch_size, self.num_frames, hidden_size)

# forward pass
with torch.no_grad():
    outputs = model(**inputs)

logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)

