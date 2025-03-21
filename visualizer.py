import cv2
from Functions3 import detection_labels
from Detectors import YOLOv10Detector
import numpy as np
import os
from PIL import Image


class Visualizer:
    def __init__(self):
        self.__video = None
        self.__video_rute = None
        self.__annotations_rute = None
        self.__detector = None

    def set_video(self, video):
        self.__video = video

    def set_video_rute(self, video_rute):
        self.__video_rute = video_rute

    def set_annotations_rute(self, annotations_rute):
        self.__annotations_rute = annotations_rute

    def set_detector(self, detector):
        self.__detector = detector

    def set_gif_making(self, gif_making):
        self.__gif_making = gif_making

    def visualize(self):
        ratio = np.array([])
        actual_rute = f"{self.__video_rute}/"
        files = os.listdir(actual_rute)
        annotations_name = files[self.__video].split(".")[0]
        annotations = np.load(f"{self.__annotations_rute}/{annotations_name}.npy")
        cap = cv2.VideoCapture(f"{self.__video_rute}/{files[self.__video]}")
        evaluate = []
        print("Here")
        i = 0
        if self.__gif_making:
            frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                """cv2.putText(
                    frame,
                    f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (172, 182, 77),
                    2,
                )"""
                if annotations[int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1]:
                    i = 0
                    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 5 == 0:
                        detections, _ = self.__detector.detection(frame)
                        printed_detections = []
                        for detection in detections:
                            if detection[1] > 0.8 and detection[0] == "person":
                                printed_detections.append(detection)
                        for detection in detections:
                            if detection[1] > 0.8 and detection[0] == "bicycle":
                                printed_detections.append(detection)
                        self.__detector.put_detections(printed_detections, frame)
                        if self.__gif_making:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(frame_rgb)
                            frames.append(pil_image)
                else:
                    i = 0
                cv2.imshow(f"{files[self.__video]}, Frame", frame)
                if cv2.waitKey(1 + i) & 0xFF == ord("q"):
                    break
                if self.__gif_making:
                    if len(frames) == 100:
                        break
            else:
                break
        cv2.destroyAllWindows()
        cap.release()
        if self.__gif_making:
            print(len(frames))
            # Step 1: Capture frames from a video or generate frames
            event = self.__video_rute.split("/")[-1]
            output_gif = f"{event}1.gif"  # Output GIF file
            # Step 2: Save frames as a GIF using PIL
            frames[0].save(
                output_gif,
                save_all=True,
                append_images=frames[1:],  # Append the rest of the frames
                duration=34,  # Delay between frames in milliseconds
                loop=0,  # Loop forever (0 means infinite loop)
            )

            print(f"GIF saved as {output_gif}")


if __name__ == "__main__":
    visualizer = Visualizer()
    events = [
        "1-Riding a bicycle",
    ]
    for i in range(len(events)):
        visualizer.set_video(0)
        rute = "../Database/CHAD DATABASE/13-Pickpockering"
        visualizer.set_video_rute(f"../Database/CHAD DATABASE/{events[i]}")
        visualizer.set_annotations_rute(
            "../Database/CHAD DATABASE/CHAD_Meta/anomaly_labels"
        )
        ov_qmodel = YOLOv10Detector()
        ov_qmodel.set_model("/home/ubuntu/yolov10/yolov10x.pt")
        ov_qmodel.set_labels(detection_labels)
        visualizer.set_detector(ov_qmodel)
        visualizer.set_gif_making(True)
        visualizer.visualize()
