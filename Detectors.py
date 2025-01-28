from abc import ABC, abstractmethod
import cv2
from ultralytics import YOLOv10


class Detectors(ABC):
    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    def set_labels(self, labels):
        pass

    @abstractmethod
    def detection(self, images):
        pass


class YOLOv10Detector(Detectors):
    def __init__(self):
        self.__model = None
        self.__labels = None

    def set_model(self, model):
        self.__model = YOLOv10(model)

    def set_labels(self, labels):
        self.__labels = labels

    def detection(self, image, classes=[]):
        detections = []
        results = self.__model(image, stream=False)
        if classes != []:
            for i in self.__labels.values():
                classes[i] = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Obtener la clase y la confianza
                class_label = int(box.cls[0])  # Convertir a entero si es necesario
                confidence = float(box.conf[0])  # Convertir a flotante si es necesario
                # Muestra la clase y el grado de confianza en el cuadro
                detections.append(
                    [self.__labels[class_label], confidence, x1, y1, x2, y2]
                )
                text = f"Class: {self.__labels[class_label] }-{confidence:.2f}, {x1}, {y1}, {x2}, {y2}"
                if classes != []:
                    classes[self.__labels[class_label]] = (
                        classes[self.__labels[class_label]] + 1
                    )
        return detections, classes

    def put_detections(self, detections, image):
        for detection in detections:
            text = f"Class: {detection[0] }-{detection[1]:.2f}, {detection[2]}, {detection[3]}, {detection[4]}, {detection[5]}"
            cv2.putText(
                image,
                text,
                (detection[2], detection[3] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 255),
                1,
            )
            cv2.rectangle(
                image,
                (detection[2], detection[3]),
                (detection[4], detection[5]),
                (255, 0, 255),
                1,
            )


if __name__ == "__main__":
    print(__name__)
