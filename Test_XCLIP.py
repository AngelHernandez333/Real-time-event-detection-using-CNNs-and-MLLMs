import torch
import numpy as np

from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download
import cv2

rute='/home/ubuntu/Database/ALL/Videos/Riding'
file='3_105_1.mp4'
cap = cv2.VideoCapture(f'{rute}/{file}')
frames=[]
while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo obtener el frame. Fin del video o error.")
        finished = True
        break
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
        frames.append(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
# Convert frames to np.ndarray of shape (num_frames, height, width, 3) and ensure RGB format
result = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])

processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
inputs = processor(
    text=["playing sports", "eating spaghetti", "go shopping"],
    videos=list(frames),
    return_tensors="pt",
    padding=True,
)

# forward pass
with torch.no_grad():
    outputs = model(**inputs)

logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)