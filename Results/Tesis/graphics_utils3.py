import cv2

cap = cv2.VideoCapture(
    "/home/ubuntu/Database/CHAD DATABASE/1-Riding a bicycle/1_066_1.mp4"
)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_num in range(0, total_frames, 6):  # Start at 0, step by 6
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        break
    print(f"Read frame {frame_num} (0-based index)")
    # Save or process the frame here

cap.release()
