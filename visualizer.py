import cv2
from ultralytics import YOLOv10
from Functions3 import detection_labels
from Detectors import YOLOv10Detector
import numpy as np


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
        annotations = np.load(f"{self.__annotations_rute}/{self.__video}.npy")
        cap = cv2.VideoCapture(f"{self.__video_rute}/{self.__video}.mp4")
        evaluate = []
        while True:
            ret, frame = cap.read()
            if ret:
                if annotations[int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1]:
                    detections, _ = self.__detector.detection(frame)
                    printed_detections = []
                    for detection in detections:
                        if detection[0] == "person" and detection[1] > 0.6:
                            printed_detections.append(detection)
                    self.__detector.put_detections(printed_detections, frame)
                    evaluate.append(printed_detections)
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.set_video("1_092_1")
    visualizer.set_video_rute("../Database/CHAD DATABASE/7-Jumping")
    visualizer.set_annotations_rute(
        "../Database/CHAD DATABASE/CHAD_Meta/anomaly_labels"
    )
    ov_qmodel = YOLOv10Detector()
    ov_qmodel.set_model("/home/ubuntu/yolov10/yolov10x.pt")
    ov_qmodel.set_labels(detection_labels)
    visualizer.set_detector(ov_qmodel)
    visualizer.visualize()
