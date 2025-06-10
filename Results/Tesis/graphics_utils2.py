import cv2
import numpy as np
from PIL import Image
import math
import os

# Function to compute destination points for perspective transformation
def get_dst_points(theta_deg, w, h, f):
    theta_rad = math.radians(theta_deg)
    s = math.sin(theta_rad)
    c = math.cos(theta_rad)
    tl_x = 0
    tl_y = -f * math.tan(theta_rad)
    tr_x = w / c if c != 0 else float('inf')
    tr_y = tl_y
    bl_x = 0
    bl_y = h * (c - s) / (s + c) if s + c != 0 else float('inf')
    br_x = w / (s + c) if s + c != 0 else float('inf')
    br_y = bl_y
    return np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)

# Parameters
video_path = '/home/ubuntu/Database/CHAD DATABASE/1-Riding a bicycle/1_066_1.mp4' # Replace with your video file path
num_frames = 30  # Number of frames to stack
tilt_increment =0 # Degrees to tilt each frame
offset_x = 100  # Horizontal offset for stacking
offset_y = 20 # Vertical offset for stacking

# Extract frames from video
cap = cv2.VideoCapture(video_path)
frames = []
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, frame_count // num_frames)
for i in range(0, frame_count, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret and len(frames) < num_frames:
        frames.append(frame)
cap.release()

# Get image dimensions
h, w = frames[0].shape[:2]
f = h  # Focal length for perspective calculation

# Initialize lists for transformed images
transformed_images = []

# Apply perspective transformation to each frame
for i in range(num_frames):
    theta = i * tilt_increment
    src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_points = get_dst_points(theta, w, h, f)
    
    # Compute bounding box for transformed image
    min_x = min(dst_points[:, 0])
    max_x = max(dst_points[:, 0])
    min_y = min(dst_points[:, 1])
    max_y = max(dst_points[:, 1])
    new_w = int(max_x - min_x)
    new_h = int(max_y - min_y)
    
    # Shift points to non-negative coordinates
    dst_points -= [min_x, min_y]
    
    # Compute perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    transformed = cv2.warpPerspective(frames[i], M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    transformed_images.append((transformed, min_x, min_y, new_w, new_h))

# Create large canvas for stacking
total_width = int(max(w, max([img[3] + img[1] + i * offset_x for i, img in enumerate(transformed_images)])))
total_height = int(max(h, max([img[4] + img[2] + i * offset_y for i, img in enumerate(transformed_images)])))
large_canvas = Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))

# Blend images with transparency
for i, (img, min_x, min_y, new_w, new_h) in enumerate(transformed_images):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    alpha = int(255 * 1)  # Decrease opacity for later frames
    img_rgba = np.dstack((img_rgb, np.full((new_h, new_w), alpha, dtype=np.uint8)))
    img_pil = Image.fromarray(img_rgba, 'RGBA')
    large_canvas.paste(img_pil, (int(i * offset_x - min_x), int(i * offset_y - min_y)), img_pil)

# Save the final image
large_canvas.save('stacked_image.png')