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

def stacked_image(frames, tilt_increment, offset_x, offset_y):
    # Get image dimensions
    num_frames = len(frames)
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
        # Add a red border to the transformed image
        border_thickness = 30  # Thickness of the red border
        scale_x = (w - 2 * border_thickness) / w
        scale_y = (h - 2 * border_thickness) / h
        scaled_transformed = cv2.resize(
            transformed,
            (int(new_w * scale_x), int(new_h * scale_y)),
            interpolation=cv2.INTER_LINEAR
        )
        transformed = cv2.copyMakeBorder(
            scaled_transformed,
            border_thickness,
            border_thickness,
            border_thickness,
            border_thickness,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 255)  # Red color in BGR
        )
        transformed = cv2.resize(
            transformed,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
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
    large_canvas.save('/home/ubuntu/Tesis/Results/Tesis/Graphics/stacked_image.png')
if __name__ == "__main__":
    # Parameters
    path = '/home/ubuntu/Tesis/Results/Tesis/Graphics/Selected_frames' # Replace with your video file path
    tilt_increment =0 # Degrees to tilt each frame
    offset_x = 120*2  # Horizontal offset for stacking
    offset_y = 30*2 # Vertical offset for stacking

    files= os.listdir(path)
    files = sorted([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    frames=[]
    for file in files:
        img = cv2.imread(os.path.join(path, file))
        frames.append(img)
    # Call the function
    stacked_image(frames, tilt_increment, offset_x, offset_y)