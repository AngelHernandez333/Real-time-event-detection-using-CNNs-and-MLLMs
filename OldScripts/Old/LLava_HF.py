import numpy as np
from huggingface_hub import hf_hub_download
import cv2
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import time


# Load the model in half-precision
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
model.to(dtype=torch.float16, device="cuda")


# Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos, up to 32 frames)
# video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")


def read_video_cv2(video_path, num_frames=8):
    """
    Decode the video with OpenCV.
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample uniformly from the video.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            break

    cap.release()
    return np.stack(frames)


# Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos, up to 32 frames)
video_path = "jobs.mp4"
video = read_video_cv2(video_path, num_frames=8)

# For videos we have to feed a "video" type instead of "image"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {
                "type": "text",
                "text": "Tell me if is true that there is a persona standing on a stage, holding a IPod Nano, the answer must be just True or False.",
            },
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to(
    "cuda:0", torch.float16
)
start_time = time.time()
out = model.generate(**inputs, max_new_tokens=60)
text_outputs = processor.batch_decode(
    out, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(text_outputs[0])

print("", time.time() - start_time, "segundos")

for i, frame in enumerate(video):
    cv2.imshow(f"Frame {i}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
