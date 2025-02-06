from abc import ABC, abstractmethod
import numpy as np

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
        if distance < max(width, height):
            return distance, True
        else:
            return distance, False
    @staticmethod
    def organize_persons(loaded_data, verbose=False):
        if verbose:
            print(
                "Informacion cargada:",
                loaded_data[0],
                "\n---------------------------------------------------------------------\n",
            )
        # Evaluate the detections and find all the persons
        persons = []
        for i in range(len(loaded_data[0])):
            if verbose:
                print(
                    "To evalute:",
                    loaded_data[0][i],
                    "\n---------------------------------------------------------------------\n",
                )
            person = [loaded_data[0][i]]
            for j in range(1, len(loaded_data)):
                stored_data_person = []
                for k in range(len(loaded_data[j])):
                    # Primeros loaded_data[0][i]
                    # Actual, de las k detecciones de los j frames
                    distance, same_person = DecisionMakerPerEvent.distance_between(
                        loaded_data[0][i], loaded_data[j][k]
                    )
                    if verbose:
                        print(
                            j,
                            "-",
                            k,
                            "-",
                            loaded_data[0][i],
                            loaded_data[j][k],
                            distance,
                            same_person,
                            "\n",
                        )
                    if same_person:
                        stored_data_person.append([loaded_data[j][k], distance])
                if len(stored_data_person) == 1:
                    loaded_data[0][i] = stored_data_person[0][0]
                elif len(stored_data_person) > 1:
                    temp = stored_data_person[0]
                    for m in range(1, len(stored_data_person)):
                        if temp[1] > stored_data_person[m][1]:
                            temp = stored_data_person[m]
                    loaded_data[0][i] = temp[0]
                person.append(loaded_data[0][i])
            persons.append(person)
        return persons
    @staticmethod
    def output_decision(condition, results, frames, MLLM):
        if condition == False and MLLM:
            frames.pop(0)
            results.pop(0)
        if condition:
            text = "yes"
        else:
            text = "no"
        return condition, text
    @staticmethod
    def verify_running(loaded_data, verbose=False):
        persons_index=[]
        for i in range(len(loaded_data)):
            area_array = np.array([])
            for j in range(len(loaded_data[i])-1):
                if verbose:
                    print(i, loaded_data[i][j], loaded_data[i][j+1], "\n")
                area =DecisionMakerPerEvent.calculate_shared_area(loaded_data[i][j], loaded_data[i][j+1])
                area_array = np.append(area_array, area)
            if verbose:
                print(area_array, area_array.mean())
            if area_array.mean() < 0.5:
                persons_index.append(i)
        return persons_index
    @staticmethod
    def verify_lying(loaded_data):
        for person in loaded_data:
            print("Lenght-",len(person), person, "\n")
        persons_index=[]
        for i in range(len(loaded_data)):
            area_array = np.array([])
            width=loaded_data[i][0][4]-loaded_data[i][0][2]
            height=loaded_data[i][0][5]-loaded_data[i][0][3]
            for j in range(len(loaded_data[i])-1):
                print(i, loaded_data[i][j], loaded_data[i][j+1], "\n")
                area =DecisionMakerPerEvent.calculate_shared_area(loaded_data[i][j], loaded_data[i][j+1])
                area_array = np.append(area_array, area)
            print(area_array, area_array.mean(), width/height)
            if area_array.mean() > 0.93 and width/height > 1:
                persons_index.append(i)
        return persons_index

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
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        persons=DecisionMakerPerEvent.verify_running(persons)
        print(len(persons) > 0, "\n---------------------------------------------------------------------\n")
        return len(persons) > 0
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
            condition, text =DecisionMakerPerEvent.output_decision(condition, results, frames, MLLM)
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
                condition, text = DecisionMakerPerEvent.output_decision(condition, results, frames, MLLM)
                return condition, text
            else:
                return False, ""
    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        persons=DecisionMakerPerEvent.verify_lying(persons)
        print(persons,len(persons) > 0, "\n---------------------------------------------------------------------\n")
        return len(persons) > 0 


class EventChasing(DecisionMakerPerEvent):
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
        if all([classes[class_] > 1 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, ""
            corrects = DecisionMakerPerEvent.check_detections(self.__classes_of_interest, detections, results)
            # Save the list to a file
            condition = self.process_detections(corrects)
            print(
                condition,
                "\n---------------------------------------------------------------------\n",
            )
            condition, text = DecisionMakerPerEvent.output_decision(condition, results, frames, MLLM)
            return condition, text
        else:
            return False, ""
    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        persons_index=DecisionMakerPerEvent.verify_running(persons)
        print(len(persons))
        if len(persons_index) <1 or len(persons) < 2:
            return False
        print('Checking persons', persons_index)
        for i in persons_index:
            for j in [x for x in range(len(persons)) if x != i]:
                print(f'Testing the person {i} with the person {j}\n')
                distance_array = np.array([])
                width = persons[i][0][4] - persons[i][0][2]
                for k in range(len(persons[i])):
                    distance, _ = DecisionMakerPerEvent.distance_between(persons[i][k],persons[j][k])
                    print(f'At frame {k} {persons[i][k]}, {persons[j][k]} {distance}\n')
                    distance_array = np.append(distance_array, distance)
                print(distance_array)
                decresing= np.all(np.diff(distance_array) < 0)
                print(decresing, width, np.std(distance_array))
                if decresing:
                    return True
                if np.std(distance_array)<width:
                    return True
        return False
        if len(persons_index) >0:
            return False
        for i in persons_index:
            print(persons[i])
            for j in [x for x in range(len(persons)) if x != i]:
                print(i, '-', j, '-',persons[j])
        # Evaluate the persons
        '''if len(persons) < 2:
            return False
        else:
            for i in range(len(persons) - 1):
                print(
                    f"Person {i}",
                    "\n---------------------------------------------------------------------\n",
                )
                for j in range(i + 1, len(persons)):
                    diferences = np.array([])
                    for k in range(len(persons[i])):
                        print(i, j, k, persons[i][k], persons[j][k], "\n")
                        distance, _ = DecisionMakerPerEvent.distance_between(persons[i][k], persons[j][k])
                        diferences = np.append(diferences, distance)
                    intervals = diferences - diferences[0]
                    print(
                        "Diferences:",
                        diferences,
                        "Intervals:",
                        intervals,
                        "Mean:",
                        intervals.mean(),
                        "\n",
                    )
                    if (
                        abs(intervals.mean()) < diferences[0] * 0.6
                        and abs(intervals.mean()) > diferences[0] * 0.01
                    ):
                        return True
                    else:
                        return False
        return False'''

class EventJumping(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest=[
            'person',]
    def detections_treatment(self):
        pass
    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, ""
            corrects = DecisionMakerPerEvent.check_detections(self.__classes_of_interest, detections, results)
            condition = self.process_detections(corrects)
            print(
                condition,
                "\n---------------------------------------------------------------------\n",
            )
            condition, text = DecisionMakerPerEvent.output_decision(condition, results, frames, MLLM)
            return condition, text
        else:
            return False, ""
    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        # Evaluate the persons
        if len(persons) < 1:
            return False
        else:
            for person in persons:
                diferences = np.array([])
                for i in range(len(person)):
                    print(i, "-", person[i][3], person[i][5], "\n")
                    diferences = np.append(diferences, person[i][5])
                intervals = diferences - diferences[0]
                height = person[0][5] - person[0][3]
                print(
                    "Intervals:",
                    intervals,
                    "Height:",
                    height,
                    "Mean:",
                    intervals.sum(),
                    "\n",
                )
                if (
                    abs(intervals.sum()) < height * 4.65
                    and abs(intervals.sum()) > height * 0.0015
                ):
                    return True
                """        if (abs(intervals.mean()) < height*0.198
                    and abs(intervals.mean()) > height*0.0015
                ):
                    return True"""
        return False

class EventFalling(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest=[
            'person',]
    def detections_treatment(self):
        pass
    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, ""
            corrects = DecisionMakerPerEvent.check_detections(self.__classes_of_interest, detections, results)
            condition = self.process_detections(corrects)
            print(
                condition,
                "\n---------------------------------------------------------------------\n",
            )
            condition, text = DecisionMakerPerEvent.output_decision(condition, results, frames, MLLM)
            return condition, text
        else:
            return False, ""
    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        # Evaluate the persons
        if len(persons) < 1:
            return False
        else:
            for person in persons:
                width_per_height_ratio = np.array([])
                diferences = np.array([])
                for i in range(len(person)):
                    print(
                        i,
                        "-",
                        person[i][2],
                        person[i][3],
                        person[i][4],
                        person[i][5],
                        "\n",
                    )
                    diferences = np.append(diferences, person[i][3])
                    height = person[i][5] - person[i][3]
                    width = person[i][4] - person[i][2]
                    width_per_height_ratio = np.append(
                        width_per_height_ratio, width / height
                    )
                print(
                    "Differences:",
                    diferences,
                )
                intervals = diferences - diferences[0]
                width_per_height_ratio = width_per_height_ratio - width_per_height_ratio[0]
                print(
                    "Intervals:",
                    intervals,
                    "Width per height ratio:",
                    width_per_height_ratio,
                    "Sum difference:",
                    intervals.sum(),
                    "Sum ratio:",
                    width_per_height_ratio.sum(),
                    "\n",
                )
                if abs(intervals.sum()) > 100 and abs(width_per_height_ratio.sum()) > 0.5:
                    return True
        return False

class EventGuiding(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest=[
            'person',]
    def detections_treatment(self):
        pass
    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, ""
            corrects = DecisionMakerPerEvent.check_detections(self.__classes_of_interest, detections, results)
            print("Correcs:", len(corrects))
            # Save the list to a file
            condition = self.process_detections(corrects)
            print(
                condition,
                "\n---------------------------------------------------------------------\n",
            )
            condition, text = DecisionMakerPerEvent.output_decision(condition, results, frames, MLLM)
            return condition, text
        else:
            return False, ""
    def process_detections(self, loaded_data, verbose=False):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        # Evaluate the persons
        if len(persons) < 2:
            return False
        else:
            check = []
            for i in range(len(persons) - 1):
                if verbose:
                    print(
                        f"Person {i}",
                        "\n---------------------------------------------------------------------\n",
                    )
                for j in range(i + 1, len(persons)):
                    touching = np.array([])
                    for k in range(len(persons[i])):
                        print(i, j, k, persons[i][k], persons[j][k], "\n")
                        touch = DecisionMakerPerEvent.boxes_touching([persons[i][k], persons[j][k]])
                        if touch:
                            touching = np.append(touching, 1)
                        else:
                            touching = np.append(touching, 0)
                    if touching.sum() > (len(touching) - 1):
                        # return True
                        check.append([persons[i], persons[j]])
                        print("This one")
            for duo in check:
                widht = duo[0][0][4] - duo[0][0][2]
                height = duo[0][0][5] - duo[0][0][3]
                distancex = np.array([])
                distancey = np.array([])
                for i in range(len(duo[0]) - 1):
                    distance1x, distance1y = DecisionMakerPerEvent.distance_direction(duo[0][i], duo[0][i + 1])
                    distance2x, distance2y = DecisionMakerPerEvent.distance_direction(duo[1][i], duo[1][i + 1])
                    distancex = np.append(distancex, abs(distance1x - distance2x))
                    distancey = np.append(distancey, abs(distance1y - distance2y))
                    print(i, "-", distance1x, distance1y, distance2x, distance2y)
                print(distancey.sum(), distancex.sum())
                print(min(widht, height))
                if distancey.sum() < height * 1 and distancex.sum() < widht * 1:
                    return True
        return False

class EventGarbage(DecisionMakerPerEvent):
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
            print("Correcs:", corrects)
            condition = self.process_detections(corrects)
            print(
                condition,
                "\n---------------------------------------------------------------------\n",
            )
            condition, text = DecisionMakerPerEvent.output_decision(condition, results, frames, MLLM)
            return condition, text
        else:
            return False, ""
    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        # Evaluate the persons
        if len(persons) < 1:
            return False
        else:
            for person in persons:
                print("Lenght-", person)
                temp=np.array([])
                for time_frame in person:
                    width = time_frame[4] - time_frame[2]
                    height = time_frame[5] - time_frame[3]
                    print(time_frame, width, height, width/height)
                    temp = np.append(temp, width / height)
                print(temp.mean(), np.abs(np.diff(temp)).mean(), temp[0]*0.15)
                if np.abs(np.diff(temp)).mean() > temp[0]*0.15:
                    return True
        return False

class EventStealing(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest=[
            'person',]
    def detections_treatment(self):
        pass
    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        pass
    def process_detections(self, loaded_data):
        pass
if __name__ == "__main__":
    pass