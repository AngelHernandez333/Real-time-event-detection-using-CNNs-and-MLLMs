from abc import ABC, abstractmethod
import numpy as np
import time


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
        stored = []
        n = len(correct)
        for i in range(n):
            for j in range(i + 1, n):
                if DecisionMakerPerEvent.do_rectangles_touch(correct[i], correct[j]):
                    stored.append(correct[i])
                    stored.append(correct[j])
        return stored

    @staticmethod
    def check_detections(classes_of_interest, detections, results):
        correct = []
        corrects = []
        for det_per_frame in results:
            for det in det_per_frame:
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
            print(
                "Reference size:",
                width,
                "Evaluate size:",
                height,
                "Distance:",
                distance,
            )
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
        persons_index = []
        for i in range(len(loaded_data)):
            area_array = np.array([])
            for j in range(len(loaded_data[i]) - 1):
                if verbose:
                    print(i, loaded_data[i][j], loaded_data[i][j + 1], "\n")
                area = DecisionMakerPerEvent.calculate_shared_area(
                    loaded_data[i][j], loaded_data[i][j + 1]
                )
                area_array = np.append(area_array, area)
            if verbose:
                print(area_array, area_array.mean())
            if area_array.mean() < 0.5:
                persons_index.append(i)
        return persons_index

    @staticmethod
    def verify_lying(loaded_data, verbose=False):
        if verbose:
            for person in loaded_data:
                print("Lenght-", len(person), person, "\n")
        persons_index = []
        for i in range(len(loaded_data)):
            area_array = np.array([])
            width = loaded_data[i][0][4] - loaded_data[i][0][2]
            height = loaded_data[i][0][5] - loaded_data[i][0][3]
            for j in range(len(loaded_data[i]) - 1):
                if verbose:
                    print(i, loaded_data[i][j], loaded_data[i][j + 1], "\n")
                area = DecisionMakerPerEvent.calculate_shared_area(
                    loaded_data[i][j], loaded_data[i][j + 1]
                )
                area_array = np.append(area_array, area)
            if verbose:
                print(area_array, area_array.mean(), width / height)
            if area_array.mean() > 0.93 and width / height > 1:
                persons_index.append(i)
        return persons_index

    @staticmethod
    def verify_falling(persons, verbose=False):
        persons_index = []
        counter = 0
        for person in persons:
            width_per_height_ratio = np.array([])
            for i in range(len(person)):
                if verbose:
                    print(
                        i,
                        "-",
                        person[i][2],
                        person[i][3],
                        person[i][4],
                        person[i][5],
                        "\n",
                    )
                height = person[i][5] - person[i][3]
                width = person[i][4] - person[i][2]
                width_per_height_ratio = np.append(
                    width_per_height_ratio, width / height
                )
            if verbose:
                print(
                    "Ratio:",
                    width_per_height_ratio,
                )
            ratio_increasing = np.all(np.diff(width_per_height_ratio) >= 0)
            if ratio_increasing:
                persons_index.append(counter)
            counter += 1
        return persons_index

    @staticmethod
    def verify_shaking(persons, verbose=False):
        persons_index = []
        counter = 0
        for person in persons:
            width_per_height_ratio = np.array([])
            for i in range(len(person)):
                if verbose:
                    print(
                        i,
                        "-",
                        person[i][2],
                        person[i][3],
                        person[i][4],
                        person[i][5],
                        "\n",
                    )
                height = person[i][5] - person[i][3]
                width = person[i][4] - person[i][2]
                width_per_height_ratio = np.append(
                    width_per_height_ratio, width / height
                )
            if verbose:
                print(
                    "Ratio:",
                    width_per_height_ratio,
                    np.abs(np.diff(width_per_height_ratio)).mean(),
                    width_per_height_ratio[0] * 0.10,
                )

            ratio_changing = (
                np.abs(np.diff(width_per_height_ratio)).mean()
                > width_per_height_ratio[0] * 0.10
            )
            if ratio_changing:
                persons_index.append(counter)
            counter += 1
        return persons_index


class EventBicycle(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = ["person", "bicycle"]

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
        stored = []
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            correct = self.detections_treatment(detections)
            stored = DecisionMakerPerEvent.boxes_touching(correct)
            decision = len(stored) > 0
            if decision:
                text = "yes"
            else:
                text = "no"
            return decision, text, stored
        else:
            return False, "", stored


class EventFight(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = ["person"]

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
        stored = []
        if all([classes[class_] > 1 for class_ in self.__classes_of_interest]):
            correct = self.detections_treatment(detections)
            stored = DecisionMakerPerEvent.boxes_touching(correct)
            decision = len(stored) > 0
            if decision:
                text = "yes"
            else:
                text = "no"
            return decision, text, stored
        else:
            return False, "", stored


class EventPlaying(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "frisbee",
            "sports ball",
            "baseball glove",
            "tennis racket",
        ]

    def detections_treatment(self, detections):
        correct = []
        for detection in detections:
            if detection[0] in self.__classes_of_interest and detection[1] > 0.5:
                correct.append(detection)

        return correct

    def process_detections(self):
        pass

    def get_classes_of_interest(self):
        return self.__classes_of_interest

    def decision_maker(self, classes, detections, *args):
        stored = []
        if (
            any([classes[class_] > 0 for class_ in self.__classes_of_interest])
            and classes["person"] > 1
        ):
            stored = self.detections_treatment(detections)
            return True, "yes", stored
        else:
            return False, "no", stored


class EventRunning(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self, detections):
        pass

    def get_classes_of_interest(self):
        return self.__classes_of_interest

    def process_detections(self, loaded_data, verbose=False):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        persons_index = DecisionMakerPerEvent.verify_running(persons)
        if verbose:
            print(
                len(persons) > 0,
                "\n---------------------------------------------------------------------\n",
            )
        stored = []
        for i in range(len(persons_index)):
            stored.append(persons[persons_index[i]][-1])
        return len(persons_index) > 0, stored

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        if False:
            print(
                len(results),
                "-",
                len(frames),
                "\n---------------------------------------------------------------------\n",
            )
        stored = []
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            condition, stored = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored
        else:
            return False, "", stored


class EventLying(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self):
        pass

    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        if False:
            print(
                len(results),
                "-",
                len(frames),
                "\n---------------------------------------------------------------------\n",
            )
        stored = []
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            # Save the list to a file
            condition, stored = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored
        else:
            return False, "", stored

    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        persons_index = DecisionMakerPerEvent.verify_lying(persons)
        if False:
            print(
                persons,
                len(persons) > 0,
                "\n---------------------------------------------------------------------\n",
            )
        stored = []
        for i in range(len(persons_index)):
            stored.append(persons[persons_index[i]][-1])
        return len(persons_index) > 0, stored


class EventChasing(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self):
        pass

    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        if False:
            print(
                len(results),
                "-",
                len(frames),
                "\n---------------------------------------------------------------------\n",
            )
        stored = []
        if all([classes[class_] > 1 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            # Save the list to a file
            condition, stored = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored
        else:
            return False, "", stored

    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        persons_index = DecisionMakerPerEvent.verify_running(persons)
        stored = []
        if len(persons_index) < 1 or len(persons) < 2:
            return False, []
        if False:
            print("Checking persons", persons_index)
        for i in persons_index:
            for j in [x for x in range(len(persons)) if x != i]:
                print(f"Testing the person {i} with the person {j}\n")
                distance_array = np.array([])
                width = persons[i][0][4] - persons[i][0][2]
                for k in range(len(persons[i])):
                    distance, _ = DecisionMakerPerEvent.distance_between(
                        persons[i][k], persons[j][k]
                    )
                    if False:
                        print(
                            f"At frame {k} {persons[i][k]}, {persons[j][k]} {distance}\n"
                        )
                    distance_array = np.append(distance_array, distance)
                if False:
                    print(distance_array)
                decresing = np.all(np.diff(distance_array) < 0)
                if False:
                    print(decresing, width, np.std(distance_array))
                if decresing or np.std(distance_array) < width:
                    stored.append(persons[i])
                    stored.append(persons[j])
        return len(stored) > 0, stored


class EventJumping(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self):
        pass

    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        stored = []
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            condition, stored = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored
        else:
            return False, "", stored

    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        # Evaluate the persons
        stored = []
        if len(persons) < 1:
            return False, stored
        else:
            for person in persons:
                diferences = np.array([])
                for i in range(len(person)):
                    if False:
                        print(i, "-", person[i][3], person[i][5], "\n")
                    diferences = np.append(diferences, person[i][5])
                height = person[0][5] - person[0][3]
                incresing = np.all(np.diff(diferences) > 0)
                decresing = np.all(np.diff(diferences) < 0)
                if False:
                    print(diferences, incresing, decresing, height)
                if incresing or decresing:
                    stored.append(person[-1])
        return len(stored) > 0, stored


class EventFalling(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self):
        pass

    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        stored = []
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            condition, stored = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored
        else:
            return False, "", stored

    def process_detections(self, loaded_data):
        stored = []
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        person_index = DecisionMakerPerEvent.verify_falling(persons, False)
        for i in person_index:
            stored.append(persons[i][-1])
        return len(person_index) > 0, stored


class EventGuiding(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self):
        pass

    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        stored = []
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            # Save the list to a file
            condition, stored = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored
        else:
            return False, "", stored

    def process_detections(self, loaded_data, verbose=False):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        # Evaluate the persons
        stored = []
        if len(persons) < 2:
            return False, stored
        else:
            # First check the couple of persons that are together
            check = []
            for i in range(len(persons) - 1):
                if verbose:
                    if False:
                        print(
                            f"Person {i}",
                            "\n---------------------------------------------------------------------\n",
                        )
                for j in range(i + 1, len(persons)):
                    touching = np.array([])
                    for k in range(len(persons[i])):
                        if False:
                            print(i, j, k, persons[i][k], persons[j][k], "\n")
                        touch = DecisionMakerPerEvent.boxes_touching(
                            [persons[i][k], persons[j][k]]
                        )
                        decision = len(touch) > 0
                        if decision:
                            touching = np.append(touching, 1)
                        else:
                            touching = np.append(touching, 0)
                    if touching.sum() > (len(touching) - 1):
                        check.append([persons[i], persons[j]])
            for duo in check:
                # Check each duo
                widht = duo[0][0][4] - duo[0][0][2]
                height = duo[0][0][5] - duo[0][0][3]
                distancex = np.array([])
                distancey = np.array([])
                for i in range(len(duo[0]) - 1):
                    distance1x, distance1y = DecisionMakerPerEvent.distance_direction(
                        duo[0][i], duo[0][i + 1]
                    )
                    distance2x, distance2y = DecisionMakerPerEvent.distance_direction(
                        duo[1][i], duo[1][i + 1]
                    )
                    distancex = np.append(distancex, abs(distance1x - distance2x))
                    distancey = np.append(distancey, abs(distance1y - distance2y))
                    if False:
                        print(i, "-", distance1x, distance1y, distance2x, distance2y)
                if False:
                    print(distancey.sum(), distancex.sum())
                    print(min(widht, height))
                if distancey.sum() < height * 1 and distancex.sum() < widht * 1:
                    stored.append(duo[0])
                    stored.append(duo[1])
                    return True, stored
        return False, stored


class EventGarbage(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self):
        pass

    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        stored = []
        if False:
            print(
                len(results),
                "-",
                len(frames),
                "\n---------------------------------------------------------------------\n",
            )
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            if False:
                print("Correcs:", corrects)
            condition, stored = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored
        else:
            return False, "", stored

    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        # Evaluate the persons
        stored = []
        if len(persons) < 1:
            return False, stored
        else:
            for person in persons:
                if False:
                    print("Lenght-", person)
                temp = np.array([])
                for time_frame in person:
                    width = time_frame[4] - time_frame[2]
                    height = time_frame[5] - time_frame[3]
                    if False:
                        print(time_frame, width, height, width / height)
                    temp = np.append(temp, width / height)
                if False:
                    print(temp.mean(), np.abs(np.diff(temp)).mean(), temp[0] * 0.15)
                if np.abs(np.diff(temp)).mean() > temp[0] * 0.15:
                    stored.append(person[-1])
                    return True, stored
        return False, stored


class EventTripping(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self):
        pass

    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        stored = []
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            condition, stored = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored
        else:
            return False, "", stored

    def process_detections(self, loaded_data, verbose=False):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        person_index = DecisionMakerPerEvent.verify_shaking(persons, verbose)
        stored = []
        if verbose:
            print("The persons shaking are", person_index, len(persons))
        if len(person_index) < 1 or len(persons) < 2:
            return False, stored
        for i in person_index:
            for j in [x for x in range(len(persons)) if x != i]:
                tripping = np.array([])
                for k in range(len(persons[i])):
                    stored = DecisionMakerPerEvent.boxes_touching(
                        [persons[i][k], persons[j][k]]
                    )
                    trip = len(stored) > 0
                    if trip:
                        tripping = np.append(tripping, 1)
                    else:
                        tripping = np.append(tripping, 0)
                if tripping.sum() > len(tripping) // 2:
                    stored.append(persons[i][-1])
                    stored.append(persons[j][-1])
        return len(stored) > 0, stored


class EventStealing(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self):
        pass

    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        pass

    def process_detections(self, loaded_data):
        pass


class EventStealing(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self):
        pass

    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        if False:
            print(
                len(results),
                "-",
                len(frames),
                "\n---------------------------------------------------------------------\n",
            )
        stored = []
        if all([classes[class_] > 1 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            # Save the list to a file
            condition, stored = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored
        else:
            return False, "", stored

    def process_detections(self, loaded_data, verbose=False):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        persons_index = DecisionMakerPerEvent.verify_running(persons)
        #print("este", persons_index)
        stored = []
        if len(persons_index) < 1 or len(persons) < 2:
            return False, stored
        if verbose:
            print("Checking persons", persons_index)
        for i in persons_index:
            for j in [x for x in range(len(persons)) if x != i]:
                if verbose:
                    print(f"Testing the person {i} with the person {j}\n")
                distance_array = np.array([])
                width = persons[i][0][4] - persons[i][0][2]
                for k in range(len(persons[i])):
                    distance, _ = DecisionMakerPerEvent.distance_between(
                        persons[i][k], persons[j][k]
                    )
                    if verbose:
                        print(
                            f"At frame {k} {persons[i][k]}, {persons[j][k]} {distance}\n"
                        )
                    distance_array = np.append(distance_array, distance)
                decresing = np.all(np.diff(distance_array) < 0)
                if verbose:
                    print(distance_array)
                    print(decresing, width, np.std(distance_array))
                if decresing or np.std(distance_array) < width:
                    for k in range(len(persons[i])):
                        if verbose:
                            print(f"At frame {k} {persons[i][k]}, {persons[j][k]}\n")
                        touching = DecisionMakerPerEvent.do_rectangles_touch(
                            persons[i][k], persons[j][k]
                        )
                        if touching:
                            stored.append(persons[i][-1])
                            stored.append(persons[j][-1])
                            return True, stored
        return False, stored


class EventPickPockering(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = [
            "person",
        ]

    def detections_treatment(self):
        pass

    def get_classes_of_interest(self):
        pass

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        if False:
            print(
                len(results),
                "-",
                len(frames),
                "\n---------------------------------------------------------------------\n",
            )
        stored = []
        if all([classes[class_] > 1 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            # Save the list to a file
            condition, stored = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored
        else:
            return False, "", stored

    def process_detections(self, loaded_data, verbose=False):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        stored = []
        # Evaluate the persons
        if len(persons) < 2:
            return False, stored
        else:
            # First check the couple of persons that are together
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
                        if verbose:
                            print(i, j, k, persons[i][k], persons[j][k], "\n")
                        stored = DecisionMakerPerEvent.boxes_touching(
                            [persons[i][k], persons[j][k]]
                        )
                        touch = len(stored) > 0
                        if touch:
                            touching = np.append(touching, 1)
                        else:
                            touching = np.append(touching, 0)
                    if touching.sum() > 0:
                        check.append([persons[i], persons[j]])
            if verbose:
                print("Status ", len(check) > 0)
            if len(check) == 0:
                return False, stored
            for duo in check:
                shaking = DecisionMakerPerEvent.verify_shaking(duo)
                if len(shaking) > 0:
                    stored.append(duo[0][-1])
                    stored.append(duo[1][-1])
                    return True, stored
        return False, stored


class ALL_Rules:
    def __init__(self):
        self.__rules = [
            EventBicycle(),
            EventFight(),
            EventPlaying(),
            EventRunning(),
            EventLying(),
            EventChasing(),
            EventJumping(),
            EventFalling(),
            EventGuiding(),
            EventStealing(),
            EventGarbage(),
            EventTripping(),
            EventPickPockering(),
        ]
        self.__descriptions = []

    def set_descriptions(self, descriptions):
        self.__descriptions = descriptions

    def get_descriptions(self):
        return self.__descriptions

    def area_torecort(self, rois):
        area_dict = {"x1": None, "y1": None, "x2": None, "y2": None}
        for roi in rois:
            print(roi)
            for i in range(1, len(roi)):
                x1, y1, x2, y2 = roi[i][2], roi[i][3], roi[i][4], roi[i][5]
                # print(x1, y1, x2, y2)
                if area_dict["x1"] is None or x1 < area_dict["x1"]:
                    area_dict["x1"] = x1
                if area_dict["y1"] is None or y1 < area_dict["y1"]:
                    area_dict["y1"] = y1
                if area_dict["x2"] is None or x2 > area_dict["x2"]:
                    area_dict["x2"] = x2
                if area_dict["y2"] is None or y2 > area_dict["y2"]:
                    area_dict["y2"] = y2
        return area_dict

    def process(self, classes, detections, results, frames, MLLM):
        prompts = []
        rois = []
        print(len(self.__descriptions), len(self.__rules))
        for i in range(len(self.__rules)):
            condition, text, objects = self.__rules[i].decision_maker(
                classes, detections, results, frames, MLLM
            )
            if condition and "" != text:
                #print("Test: ", self.__descriptions[i], objects)
                prompts.append(self.__descriptions[i])
                rois.append(objects)
        #to_recort = self.area_torecort(rois)
        return prompts


if __name__ == "__main__":
    pass
