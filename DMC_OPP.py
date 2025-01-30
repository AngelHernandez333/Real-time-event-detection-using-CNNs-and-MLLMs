from abc import ABC, abstractmethod

class DecisionMakerPerEvent(ABC):
    @abstractmethod
    def detections_treatment(self):
        pass
    @abstractmethod
    def get_classes_of_interest(self):
        pass
    @abstractmethod
    def decision_maker(self):
        pass
    @abstractmethod
    def process_detections(self):
        pass
    @staticmethod
    def identify_persons():
        pass
    @staticmethod
    def do_rectangles_touch(rect1, rect2):
        x1_1, y1_1, x2_1, y2_1 = rect1[2], rect1[3], rect1[4], rect1[5]
        x1_2, y1_2, x2_2, y2_2 = rect2[2], rect2[3], rect2[4], rect2[5]
        # Check if the rectangles overlap horizontally and vertically
        horizontal_overlap = not (x2_1 < x1_2 or x2_2 < x1_1)
        vertical_overlap = not (y2_1 < y1_2 or y2_2 < y1_1)
        return horizontal_overlap and vertical_overlap
    @staticmethod
    def boxes_touching(correct):
        n = len(correct)
        for i in range(n):
            for j in range(i + 1, n):
                if DecisionMakerPerEvent.do_rectangles_touch(correct[i], correct[j]):
                    return True
        return False
    @staticmethod
    def check_detections(classes_of_interest, detections, results):
        correct = []
        corrects = []
        for det_per_frame in results:
            for det in det_per_frame :
                if det[0] == classes_of_interest[0] and det[1] > 0.7:
                    correct.append(det)
            corrects.append(correct)
            correct = []
        for detection in detections:
            if detection[0] == classes_of_interest[0] and detection[1] > 0.7:
                correct.append(detection)
        corrects.append(correct)
        return corrects
    @staticmethod
    def calculate_shared_area(rect1, rect2):
        x1_1, y1_1, x2_1, y2_1 = rect1[2], rect1[3], rect1[4], rect1[5]
        x1_2, y1_2, x2_2, y2_2 = rect2[2], rect2[3], rect2[4], rect2[5]
        # Calculate the coordinates of the intersection rectangle
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        # Calculate the width and height of the intersection rectangle
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        area_rect1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        # Calculate the area of the intersection rectangle
        inter_area = inter_width * inter_height
        return inter_area / area_rect1
    @staticmethod
    def distance_direction(reference, evaluate, verbose=False):
        x1_1, y1_1, x2_1, y2_1 = reference[2], reference[3], reference[4], reference[5]
        x1_2, y1_2, x2_2, y2_2 = evaluate[2], evaluate[3], evaluate[4], evaluate[5]

        # Calculate the centroids of the bounding boxes
        centroid_ref = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        centroid_eval = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)

        return centroid_ref[0] - centroid_eval[0], centroid_ref[1] - centroid_eval[1]

    @staticmethod
    def distance_between(reference, evaluate, verbose=False):
        x1_1, y1_1, x2_1, y2_1 = reference[2], reference[3], reference[4], reference[5]
        x1_2, y1_2, x2_2, y2_2 = evaluate[2], evaluate[3], evaluate[4], evaluate[5]

        # Calculate the centroids of the bounding boxes
        centroid_ref = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        centroid_eval = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)

        # Calculate the distance between the centroids
        distance = (
            (centroid_ref[0] - centroid_eval[0]) ** 2
            + (centroid_ref[1] - centroid_eval[1]) ** 2
        ) ** 0.5
        width = x2_1 - x1_1
        height = y2_1 - y1_1
        if verbose:
            print("Reference size:", width, "Evaluate size:", height, "Distance:", distance)
        if distance < min(width, height):
            return distance, True
        else:
            return distance, False


class EventBicycle(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest=["person", "bicycle"]
    def detections_treatment(self, detections):
        correct = []
        for detection in detections:
            if detection[0] in self.__classes_of_interest and detection[1] > 0.5:
                correct.append(detection)
        return correct
    def get_classes_of_interest(self):
        return self.__classes_of_interest
    def process_detections(self):
        pass
    def decision_maker(self, classes, detections, *args):
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            correct=self.detections_treatment(detections)
            decision = DecisionMakerPerEvent.boxes_touching(correct)
            if decision:
                text = "yes"
            else:
                text = "no"
            return decision, text
        else:
            return False, ""

class EventFight(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest=["person"]
    def detections_treatment(self, detections):
        correct = []
        for detection in detections:
            if detection[0] in self.__classes_of_interest and detection[1] > 0.5:
                correct.append(detection)
        return correct
    def get_classes_of_interest(self):
        return self.__classes_of_interest
    def process_detections(self):
        pass
    def decision_maker(self, classes, detections, *args):
        if all([classes[class_] > 1 for class_ in self.__classes_of_interest]):
            correct=self.detections_treatment(detections)
            decision = DecisionMakerPerEvent.boxes_touching(correct)
            if decision:
                text = "yes"
            else:
                text = "no"
            return decision, text
        else:
            return False, ""

class EventPlaying(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest=[
        "frisbee",
        "sports ball",
        "baseball glove",
        "tennis racket",
    ]
    def detections_treatment(self, detections):
        pass
    def process_detections(self):
        pass
    def get_classes_of_interest(self):
        return self.__classes_of_interest
    def decision_maker(self, classes, detections, *args):
        if (
            any([classes[class_] > 0 for class_ in self.__classes_of_interest])
            and classes["person"] > 1
        ):
            return True, "yes"
        else:
            return False, "no"

class EventRunning(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest=[
        'person',
    ]
    def detections_treatment(self, detections):
        pass
    def get_classes_of_interest(self):
        return self.__classes_of_interest
    def process_detections(self, loaded_data):
        for i in range(len(loaded_data[0])):
            print(
                "To evalute:",
                loaded_data[0][i],
                "\n---------------------------------------------------------------------\n",
            )
            contador = 0
            for j in range(1, len(loaded_data)):
                stored_data = []
                for k in range(len(loaded_data[j])):
                    print(
                        j,
                        "-",
                        k,
                        "-",
                        loaded_data[0][i],
                        loaded_data[j][k],
                        DecisionMakerPerEvent.calculate_shared_area(loaded_data[0][i], loaded_data[j][k]),
                        "\n",
                    )
                    if (
                        DecisionMakerPerEvent.calculate_shared_area(loaded_data[0][i], loaded_data[j][k]) > 0.2
                        and DecisionMakerPerEvent.calculate_shared_area(loaded_data[0][i], loaded_data[j][k])
                        < 0.9
                    ):
                        if stored_data == []:
                            stored_data.append(
                                [
                                    loaded_data[j][k],
                                    DecisionMakerPerEvent.calculate_shared_area(
                                        loaded_data[0][i], loaded_data[j][k]
                                    ),
                                ]
                            )
                        else:
                            if stored_data[0][1] < DecisionMakerPerEvent.calculate_shared_area(
                                loaded_data[0][i], loaded_data[j][k]
                            ):
                                stored_data[0] = [
                                    loaded_data[j][k],
                                    DecisionMakerPerEvent.calculate_shared_area(
                                        loaded_data[0][i], loaded_data[j][k]
                                    ),
                                ]
                if len(stored_data) > 0:
                    contador += 1
                    loaded_data[0][i] = stored_data[0][0]
            print(contador, "----")
            if contador > 3:
                return True
        return False
    
    def decision_maker(self, classes, detections,results, frames,MLLM, *args):
        print(
            len(results),
            "-",
            len(frames),
            "\n---------------------------------------------------------------------\n",
        )
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, ""
            corrects = DecisionMakerPerEvent.check_detections(self.__classes_of_interest, detections, results)
            condition = self.process_detections(corrects)
            print(
                condition,
                "\n---------------------------------------------------------------------\n",
            )
            if condition == False and MLLM:
                frames.pop(0)
                results.pop(0)
            if condition:
                text = "yes"
            else:
                text = "no"
            return condition, text
        else:
            return False, ""

class EventLying(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest=[
        'person',]
    def detections_treatment(self):
        pass
    def get_classes_of_interest(self):
        pass
    def decision_maker(self, classes, detections, results, frames, MLLM, *args):

            print(
                len(results),
                "-",
                len(frames),
                "\n---------------------------------------------------------------------\n",
            )
            if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
                if len(results) < 6:
                    return True, ""
                corrects = DecisionMakerPerEvent.check_detections(self.__classes_of_interest, detections, results)
                # Save the list to a file
                condition = self.process_detections(corrects)
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
                if condition == False and MLLM:
                    frames.pop(0)
                    results.pop(0)
                if condition:
                    text = "yes"
                else:
                    text = "no"
                    results.pop(0)
                return condition, text
            else:
                return False, ""
    def process_detections(self):
        pass

if __name__ == "__main__":
    pass