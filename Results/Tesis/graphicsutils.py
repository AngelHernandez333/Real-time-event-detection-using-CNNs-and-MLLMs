import cv2
import os


def extract_frames(video_path, output_folder, gap, end_frame, start_frame=0):
    end_frame = start_frame + end_frame
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(
        cv2.CAP_PROP_POS_FRAMES, start_frame
    )  # Set the video to start at the first frame
    print(f"Total frames in video: {total_frames}")
    os.makedirs(output_folder, exist_ok=True)
    while True:
        ret, frame = cap.read()
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret or current_frame > end_frame:
            break
        if current_frame % gap == 0:
            cv2.imwrite(f"{output_folder}/{gap}_frame{current_frame}.jpg", frame)
            print(f"Extracted frame {current_frame}")
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
    cap.release()


if __name__ == "__main__":
    video_path = "/home/ubuntu/Database/CHAD DATABASE/1-Riding a bicycle/1_066_1.mp4"
    output_folder = "/home/ubuntu/Tesis/Results/Tesis/Graphics/Selected_frames"
    # extract_frames(video_path, output_folder, 1, 31, 200)
    extract_frames(video_path, output_folder, 5, 30, 200)
