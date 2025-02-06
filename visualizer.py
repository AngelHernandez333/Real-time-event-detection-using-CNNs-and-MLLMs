import cv2
from ultralytics import YOLOv10
from Functions3 import detection_labels
from Detectors import YOLOv10Detector
import numpy as np
import os


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

    def visualize(self):
        ratio = np.array([])
        actual_rute = f"{self.__video_rute}/"
        files = os.listdir(actual_rute)
        annotations_name = files[self.__video].split(".")[0]
        annotations = np.load(f"{self.__annotations_rute}/{annotations_name}.npy")
        cap = cv2.VideoCapture(f"{self.__video_rute}/{files[self.__video]}")
        evaluate = []
        print("Here")
        i=0
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.putText(
                    frame,
                    f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (172, 182, 77),
                    2,
                )
                if annotations[int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1]:
                    i=50
                    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 5 == 0:
                        cv2
                        detections, _ = self.__detector.detection(frame)
                        printed_detections = []
                        for detection in detections:
                            if detection[1] > 0.8 and detection[0] == "person":
                                printed_detections.append(detection)
                                print(
                                    "Frame",
                                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                                    " ",
                                    detection[4] - detection[2],
                                    detection[5] - detection[3],
                                    "Ratio ",
                                    (detection[4] - detection[2])
                                    / (detection[5] - detection[3]),
                                )
                                ratio = np.append(
                                    ratio,
                                    (detection[4] - detection[2])
                                    / (detection[5] - detection[3]),
                                )
                        self.__detector.put_detections(printed_detections, frame)
                    # evaluate.append(printed_detections)
                else:
                    i=0
                cv2.imshow(f"{files[self.__video]}, Frame", frame)
                if cv2.waitKey(1+i) & 0xFF == ord("q"):
                    break
            else:
                break
        cap.release()

if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.set_video(4)
    visualizer.set_video_rute("../Database/CHAD DATABASE/6-Chasing")
    visualizer.set_annotations_rute(
        "../Database/CHAD DATABASE/CHAD_Meta/anomaly_labels"
    )
    ov_qmodel = YOLOv10Detector()
    ov_qmodel.set_model("/home/ubuntu/yolov10/yolov10x.pt")
    ov_qmodel.set_labels(detection_labels)
    visualizer.set_detector(ov_qmodel)
    visualizer.visualize()
