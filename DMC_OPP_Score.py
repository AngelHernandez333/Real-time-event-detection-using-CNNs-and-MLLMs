from abc import ABC, abstractmethod
import numpy as np
import time
from scipy.stats import linregress

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
                if det[0] == classes_of_interest[0] and det[1] > 0.2:
                    correct.append(det)
            corrects.append(correct)
            correct = []
        for detection in detections:
            if detection[0] == classes_of_interest[0] and detection[1] > 0.2:
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
        score_actual=0
        scores = []
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
            #print(area_array, area_array.mean())
            if area_array.mean() < 0.9:
                persons_index.append(i)
                score= ((1-area_array.mean()) * loaded_data[i][-1][1])
                scores.append(score)
                if score > score_actual:
                    score_actual = score
                    rute_stored='/home/ubuntu/Tesis'
                    file='IOUS_Running.npy'
                    try:
                        ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                    except:
                        ious = np.empty((0, 3), dtype=np.float32) 
                    new_row = np.array([[area_array.mean(), loaded_data[i][-1][1], 1]], dtype=ious.dtype)
                    ious = np.vstack((ious, new_row))
                    np.save(f"{rute_stored}/{file}", ious)

        #print(f"Persons running: {persons_index}")
        return persons_index, score_actual, scores

    @staticmethod
    def verify_lying(loaded_data, verbose=False):
        if verbose:
            for person in loaded_data:
                print("Lenght-", len(person), person, "\n")
        persons_index = []
        score_actual = 0
        for i in range(len(loaded_data)):
            score=0
            rute_stored='/home/ubuntu/Tesis'
            file='IOUS_Lying.npy'
            try:
                ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
            except:
                ious = np.empty((0, 3), dtype=np.float32) 
            area_array = np.array([])
            width = loaded_data[i][-1][4] - loaded_data[i][-1][2]
            height = loaded_data[i][-1][5] - loaded_data[i][-1][3]
            for j in range(len(loaded_data[i]) - 1):
                if verbose:
                    print(i, loaded_data[i][j], loaded_data[i][j + 1], "\n")
                area = DecisionMakerPerEvent.calculate_shared_area(
                    loaded_data[i][j], loaded_data[i][j + 1]
                )
                area_array = np.append(area_array, area)
            if verbose:
            
                print(area_array, area_array.mean(), width / height)
            if width/height > 1:
                score=area_array.mean() *loaded_data[i][-1][1]
            else:
                score=area_array.mean() *loaded_data[i][-1][1]*( width / height)
            persons_index.append(i)
            if score > score_actual:
                score_actual = score
                new_row = np.array([[area_array.mean(), loaded_data[i][-1][1], width / height]], dtype=ious.dtype)
                ious = np.vstack((ious, new_row))
                np.save(f"{rute_stored}/{file}", ious)
        return persons_index, score_actual
    
    @staticmethod
    def verify_falling(persons, verbose=False):
        persons_index = []
        counter = 0
        scores=[0]
        rute_stored='/home/ubuntu/Tesis'
        file='IOUS_Falling.npy'
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
                try:
                    ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                except:
                    ious = np.empty((0, 3), dtype=np.float32) 
                persons_index.append(counter)
                scores.append(
                4*np.abs(np.diff(width_per_height_ratio)).mean()*person[-1][1])
                new_row = np.array([[np.abs(np.diff(width_per_height_ratio)).mean(), np.abs(np.diff(width_per_height_ratio)).sum(), person[-1][1]]], dtype=ious.dtype)
                ious = np.vstack((ious, new_row))
                np.save(f"{rute_stored}/{file}", ious)
            counter += 1
        return persons_index, scores

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
            if detection[0] in self.__classes_of_interest and detection[1] > 0.2:
                correct.append(detection)
        return correct

    def get_classes_of_interest(self):
        return self.__classes_of_interest

    def process_detections(self):
        pass

    def decision_maker(self, classes, detections, *args):
        stored = []
        score_actual=0
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            correct = self.detections_treatment(detections)
            stored = DecisionMakerPerEvent.boxes_touching(correct)
            decision = len(stored) > 0
            if decision:
                for i in range(0,len(stored),2):
                    if (stored[i][0] == "bicycle" and stored[i+1][0] == "person") or ((stored[i+1][0] == "bicycle" and stored[i][0] == "person") ):
                        rute_stored='/home/ubuntu/Tesis'
                        file='IOUS_Bicycles.npy'
                        try:
                            ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                        except:
                            ious = np.array([], dtype=object)
                        iou = DecisionMakerPerEvent.calculate_shared_area(stored[i], stored[i+1])
                        #print(stored[i] , stored[i+1], iou)

                        score=min(1, (stored[i][1] * stored[i+1][1])*(2*iou))
                        ious=np.append(ious, iou)
                        np.save(f"{rute_stored}/{file}", ious)
                        if score> score_actual:
                            score_actual=score
                        
            else:
                score=0
            #print(f'Score: {score_actual}\n\n')
            if decision:
                text = "yes"
            else:
                text = "no"
            return decision, text, stored, score_actual
        else:
            return False, "", stored, score_actual


class EventFight(DecisionMakerPerEvent):
    def __init__(self):
        self.__classes_of_interest = ["person"]

    def detections_treatment(self, detections):
        correct = []
        for detection in detections:
            if detection[0] in self.__classes_of_interest and detection[1] > 0.2:
                correct.append(detection)
        return correct

    def get_classes_of_interest(self):
        return self.__classes_of_interest

    def process_detections(self):
        pass

    def decision_maker(self, classes, detections, *args):
        stored = []
        score_actual = 0
        if all([classes[class_] > 1 for class_ in self.__classes_of_interest]):
            correct = self.detections_treatment(detections)
            stored = DecisionMakerPerEvent.boxes_touching(correct)
            decision = len(stored) > 0
            if decision:
                for i in range(0,len(stored),2):
                    rute_stored='/home/ubuntu/Tesis'
                    file='IOUS_Fight.npy'
                    try:
                        ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                    except:
                        ious = np.array([], dtype=object)
                    iou = DecisionMakerPerEvent.calculate_shared_area(stored[i], stored[i+1])
                    #print(stored[i] , stored[i+1], iou)
                    score=(stored[i][1] * stored[i+1][1]*iou)
                    ious=np.append(ious, iou)
                    np.save(f"{rute_stored}/{file}", ious)
                    if score> score_actual:
                        score_actual=score
            #print(f'Score: {score_actual}\n\n')
            if decision:
                text = "yes"
            else:
                text = "no"
            return decision, text, stored, score_actual
        else:
            return False, "", stored, score_actual


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
        persons=[]
        for detection in detections:
            if detection[0] in self.__classes_of_interest and detection[1] > 0.2:
                correct.append(detection)
            if detection[0] == 'person' and detection[1] > 0.2:
                persons.append(detection)

        return correct, persons

    def process_detections(self):
        pass

    def get_classes_of_interest(self):
        return self.__classes_of_interest

    def decision_maker(self, classes, detections, *args):
        stored = []
        score_actual = 0
        if (
            any([classes[class_] > 0 for class_ in self.__classes_of_interest])
            and classes["person"] > 1
        ):
            stored, persons = self.detections_treatment(detections)
            for i in range(len(stored)):
                index=None
                distance_min=None
                width=None
                score=0
                for j in range(len(persons)):
                    x,y=DecisionMakerPerEvent.distance_direction(stored[i], persons[j])
                    distance= (x**2 + y**2)**0.5
                    if distance_min is None or distance < distance_min:
                        index=j
                        distance_min=distance
                        width = persons[j][4] - persons[j][2]
                if index is None:
                    continue
                rute_stored='/home/ubuntu/Tesis'
                file='IOUS_Playing.npy'
                try:
                    ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                except:
                    ious = np.empty((0, 3), dtype=np.float32) 
                    #ious = np.array([], dtype=object)
                if distance_min< width:
                    score = stored[i][1] * persons[index][1]
                else:
                    score = stored[i][1] * persons[index][1]*(width/distance_min)
                if score > score_actual:
                    score_actual = score
                new_row = np.array([[distance_min, width, score_actual]], dtype=ious.dtype)
                ious = np.vstack((ious, new_row))
                np.save(f"{rute_stored}/{file}", ious)
            #print(f'Score: {score_actual}\n\n')
            return True, "yes", stored, score_actual
        else:
            return False, "no", stored, score_actual


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
        persons_index, score, scores = DecisionMakerPerEvent.verify_running(persons)
        if verbose:
            print(
                len(persons) > 0,
                "\n---------------------------------------------------------------------\n",
            )
        stored = []
        for i in range(len(persons_index)):
            stored.append(persons[persons_index[i]])
        return len(persons_index) > 0, stored, score

    def decision_maker(self, classes, detections, results, frames, MLLM, *args):
        if False:
            print(
                len(results),
                "-",
                len(frames),
                "\n---------------------------------------------------------------------\n",
            )
        stored = []
        score_actual=0
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored, score_actual
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            condition, stored, score = self.process_detections(corrects)

            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            if score > score_actual:
                score_actual = score
            #print(f'Score: {score_actual}\n\n')
            return condition, text, stored,score_actual
        else:
            return False, "", stored,score_actual


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
        score_actual=0
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored,score_actual
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            # Save the list to a file
            condition, stored, score = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            if score > score_actual:
                score_actual = score
            #print(f'Score: {score_actual}\n\n')
            return condition, text, stored, score_actual
        else:
            return False, "", stored, score_actual

    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        persons_index, score_actual = DecisionMakerPerEvent.verify_lying(persons)
        if False:
            print(
                persons,
                len(persons) > 0,
                "\n---------------------------------------------------------------------\n",
            )
        stored = []
        for i in range(len(persons_index)):
            stored.append(persons[persons_index[i]])
        return len(persons_index) > 0, stored, score_actual


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
        score_actual=0
        if all([classes[class_] > 1 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored, score_actual
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            # Save the list to a file
            condition, stored, score = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            score_actual=max(score, score_actual)
            #print(f'Score: {score_actual}\n\n')
            return condition, text, stored, score_actual
        else:
            return False, "", stored, score_actual

    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        persons_index, _, scores = DecisionMakerPerEvent.verify_running(persons)
        stored = []
        score_actual = 0


        def confidence_chasing_score(distance_array, running_score, width_person, weights=[0.4, 0.3, 0.3]):
            """
            Calcula el score de confianza (0-1) de que una persona esté persiguiendo a otra.
            
            Args:
                distance_array (list/array): Array de distancias entre ambas personas.
                running_score (float): Score de correr (0-1) de la persona perseguidora.
                width_person (float): Ancho estimado de la persona (para normalizar std).
                weights (list): Pesos [weight_trend, weight_std, weight_running].
                
            Returns:
                float: Score de confianza (0-1).
            """
            # 1. Score de tendencia decreciente (linregress)
            x = np.arange(len(distance_array))
            slope, _, _, _, _ = linregress(x, distance_array)
            trend_score = max(0, -slope * 0.5)  # Slope negativo -> tendencia decreciente
            trend_score = min(1, trend_score)   # Asegurar [0, 1]
            
            # 2. Score de desviación estándar (normalizada con el ancho de la persona)
            std = np.std(distance_array)
            std_score = 1 - min(1, std / width_person)  # std pequeña -> score alto
            
            # 3. Score de correr (ya definido)
            running_score = running_score  # Asumimos que ya está en [0, 1]
            
            # Combinación ponderada
            score = (weights[0] * trend_score) + (weights[1] * std_score) + (weights[2] * running_score)
            
            return score
        if len(persons_index) < 1 or len(persons) < 2:
            return False, [], score_actual
        if False:
            print("Checking persons", persons_index)
        index_score = 0
        for i in persons_index:
            for j in [x for x in range(len(persons)) if x != i]:
                #print(f"Testing the person {i} with the person {j}\n")
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
                    score=confidence_chasing_score(distance_array, scores[index_score], width, weights=[0.4, 0.3, 0.3])
                    rute_stored='/home/ubuntu/Tesis'
                    file='IOUS_Chasing.npy'
                    try:
                        ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                    except:
                        ious = np.empty((0, 4), dtype=np.float32)
                    # Use score instead of scores[i] to avoid IndexError
                    new_row = np.array([[np.abs(np.diff(distance_array).mean()), (distance_array-distance_array[0]).mean(), width, score]], dtype=ious.dtype)
                    ious = np.vstack((ious, new_row))
                    np.save(f"{rute_stored}/{file}", ious)
                    score_actual = max(score, score_actual)
            index_score += 1
        return len(stored) > 0, stored, score_actual


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
        score_actual=0
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored, score_actual
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            condition, stored, scores = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            if len(scores) > 0:
                score_actual = max(scores)
            #print(f'Score: {score_actual}\n\n')
            return condition, text, stored, score_actual
        else:
            return False, "", stored, score_actual

    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        # Evaluate the persons
        stored = []
        scores=[0]
        rute_stored='/home/ubuntu/Tesis'
        file='IOUS_Jumping.npy'
        if len(persons) < 1:
            return False, stored, [0]
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
                    stored.append(person)
                    scores.append(person[-1][1]*min(np.abs(np.diff(diferences)).sum()/(height*0.5), 1))
                    try:
                        ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                    except:
                        ious = np.empty((0, 3), dtype=np.float32) 
                    new_row = np.array([[np.abs(np.diff(diferences)).sum(), height, person[-1][1]]], dtype=ious.dtype)
                    ious = np.vstack((ious, new_row))
                    np.save(f"{rute_stored}/{file}", ious)
        return len(stored) > 0, stored, scores


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
        score_actual=0
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored, score_actual
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            condition, stored, score = self.process_detections(corrects)
            #print(stored)
            if len(score) > 0:
                score_actual = max(score)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            #print(f'Score: {score_actual}\n\n')
            return condition, text, stored, score_actual
        else:
            return False, "", stored, score_actual

    def process_detections(self, loaded_data):
        stored = []
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        person_index, scores = DecisionMakerPerEvent.verify_falling(persons, False)
        for i in person_index:
            stored.append(persons[i])
        return len(person_index) > 0, stored, scores


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
        score_actual=0
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored, score_actual
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            # Save the list to a file
            condition, stored, scores = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            if len(scores) > 0:
                score_actual = max(scores)
            #print(f'Score: {score_actual}\n\n')
            return condition, text, stored, score_actual
        else:
            return False, "", stored, score_actual

    def process_detections(self, loaded_data, verbose=False):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        # Evaluate the persons
        stored = []
        scores=[0]
        if len(persons) < 2:
            return False, stored, scores
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
                    score=min(1,
                        (distancey.sum()/height + distancex.sum()/widht)*duo[0][-1][1]*duo[1][-1][1] )
                    scores.append(score)
                    rute_stored='/home/ubuntu/Tesis'
                    file='IOUS_Guiding.npy'
                    try:
                        ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                    except:
                        ious = np.empty((0, 3), dtype=np.float32) 
                    new_row = np.array([[distancey.sum()/height,distancex.sum()/widht, duo[0][-1][1]*duo[1][-1][1]]], dtype=ious.dtype)
                    ious = np.vstack((ious, new_row))
                    np.save(f"{rute_stored}/{file}", ious)
                    return True, stored, scores
        return False, stored, scores


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
        score_actual=0
        if False:
            print(
                len(results),
                "-",
                len(frames),
                "\n---------------------------------------------------------------------\n",
            )
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored, score_actual
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            if False:
                print("Correcs:", corrects)
            condition, stored, score = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            if score!=0:
                score_actual = score
            #print(f'Scores: {score_actual}\n\n')
            return condition, text, stored, score_actual
        else:
            return False, "", stored, score_actual

    def process_detections(self, loaded_data):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        # Evaluate the persons
        stored = []
        score_actual= 0
        if len(persons) < 1:
            return False, stored, score_actual
        else:
            rute_stored='/home/ubuntu/Tesis'
            file='IOUS_Littering.npy'
            for person in persons:
                if False:
                    print("Lenght-", person)
                temp = np.array([])
                score=0
                for time_frame in person:
                    width = time_frame[4] - time_frame[2]
                    height = time_frame[5] - time_frame[3]
                    if False:
                        print(time_frame, width, height, width / height)
                    temp = np.append(temp, width / height)
                if False:
                    print(temp.mean(), np.abs(np.diff(temp)).mean(), temp[0] * 0.15)
                
                #if np.abs(np.diff(temp)).mean() > temp[0] * 0.15:
                stored.append(person)
                try:
                    ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                except:
                    ious = np.empty((0, 3), dtype=np.float32) 
                score=2*np.abs(np.diff(temp)).mean()*person[-1][1]
                if score > score_actual:
                    score_actual = score
                    new_row = np.array([[np.abs(np.diff(temp)).mean(), np.abs(np.diff(temp)).sum(), person[-1][1]]], dtype=ious.dtype)
                    ious = np.vstack((ious, new_row))
                    np.save(f"{rute_stored}/{file}", ious)
            return True, stored, score_actual
        return False, stored, score_actual


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
        score_actual=0
        if all([classes[class_] > 0 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored, score_actual
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            condition, stored, score = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            if score != 0:
                score_actual = score
            #print(f'Score: {score_actual}\n\n')
            return condition, text, stored, score_actual
        else:
            return False, "", stored, score_actual

    def process_detections(self, loaded_data, verbose=False):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        person_index = DecisionMakerPerEvent.verify_shaking(persons, verbose)
        stored = []
        score_actual=0
        rute_stored='/home/ubuntu/Tesis'
        file='IOUS_Tripping.npy'
        if verbose:
            print("The persons shaking are", person_index, len(persons))
        if len(person_index) < 1 or len(persons) < 2:
            return False, stored, score_actual
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
                if tripping.sum() > 0:
                    stored.append(persons[i])
                    stored.append(persons[j])
                    distance=DecisionMakerPerEvent.distance_between(
                        persons[i][-1], persons[j][-1]
                    )[0]
                    try:
                        ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                    except:
                        ious = np.empty((0, 3), dtype=np.float32) 
                    score=tripping.mean()* persons[i][-1][1]
                    score_actual = max(score, score_actual)
                    new_row = np.array([[tripping.sum(), persons[i][-1][4]-persons[i][-1][2], persons[i][-1][1]]], dtype=ious.dtype)
                    ious = np.vstack((ious, new_row))
                    np.save(f"{rute_stored}/{file}", ious)
        return len(stored) > 0, stored, score_actual


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
        score_actual=0
        if all([classes[class_] > 1 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored,score_actual
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            # Save the list to a file
            condition, stored, score = self.process_detections(corrects)
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            if score!= 0:
                score_actual = score
                #print(f'Score: {score_actual}\n\n')
            return condition, text, stored, score_actual
        else:
            return False, "", stored, score_actual

    def process_detections(self, loaded_data, verbose=False):
        def confidence_stealing_score(
            distance_array,
            running_score,
            width_person,
            touch_array=None,
            weights=[0.35, 0.25, 0.15, 0.25],
        ):
            """
            Calcula el score de confianza (0-1) de que una persona esté persiguiendo a otra, incluyendo si la alcanza.
            
            Args:
                distance_array (list/array): Distancias entre ambas personas a lo largo del tiempo.
                running_score (float): Score de correr (0-1) de la persona perseguidora.
                width_person (float): Ancho estimado de la persona (para normalizar std).
                weights (list): Pesos [trend, std, running, touching].
                touch_array (list[int] o None): Lista binaria (0 o 1) indicando si hubo contacto por frame.
                
            Returns:
                float: Score de confianza (0-1).
            """
            rute_stored='/home/ubuntu/Tesis'
            file='IOUS_Stealing.npy'
            # 1. Score de tendencia decreciente (linregress)
            x = np.arange(len(distance_array))
            slope, _, _, _, _ = linregress(x, distance_array)
            trend_score = max(0, -slope * 0.5)
            trend_score = min(1, trend_score)

            # 2. Score de desviación estándar (normalizada con el ancho de la persona)
            std = np.std(distance_array)
            std_score = 1 - min(1, std / width_person)

            # 3. Score de correr (ya dado)
            running_score = np.clip(running_score, 0, 1)

            # 4. Score de contacto
            if touch_array is not None and len(touch_array) == len(distance_array):
                contact_ratio = sum(touch_array) / len(touch_array)
                touch_score = min(1, contact_ratio / 0.5)  # 50% contacto = 1.0, menos = proporcional
            else:
                touch_score = 0  # No hay evidencia de contacto
            try:
                ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
            except:
                ious = np.empty((0, 4), dtype=np.float32)
            # Save the scores to a file
            new_row = np.array([[trend_score, std_score,
                                running_score, touch_score]], dtype=ious.dtype)
            ious = np.vstack((ious, new_row))
            np.save(f"{rute_stored}/{file}", ious)
            # Score total
            score = (
                weights[0] * trend_score +
                weights[1] * std_score +
                weights[2] * running_score +
                weights[3] * touch_score
            )
            
            return score
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        persons_index, _, scores = DecisionMakerPerEvent.verify_running(persons)
        #print("este", persons_index)
        stored = []
        score_actual=0
        index_score = 0
        if len(persons_index) < 1 or len(persons) < 2:
            return False, stored, score_actual
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
                    touching_array = np.array([])
                    for k in range(len(persons[i])):
                        if verbose:
                            print(f"At frame {k} {persons[i][k]}, {persons[j][k]}\n")
                        touching = DecisionMakerPerEvent.do_rectangles_touch(
                            persons[i][k], persons[j][k]
                        )
                        touching_array = np.append(touching_array, touching)
                    score= confidence_stealing_score(
                                distance_array,
                                scores[index_score],
                                width,
                                touch_array=touching_array,
                            )
                    score_actual = max(score, score_actual)
            index_score += 1
        return False, stored, score_actual


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
        score_actual= 0
        if all([classes[class_] > 1 for class_ in self.__classes_of_interest]):
            if len(results) < 6:
                return True, "", stored, score_actual
            corrects = DecisionMakerPerEvent.check_detections(
                self.__classes_of_interest, detections, results
            )
            # Save the list to a file
            condition, stored, score = self.process_detections(corrects)
            if score != 0:
                score_actual = score
            #print(f'Score {score_actual}\n\n')
            if False:
                print(
                    condition,
                    "\n---------------------------------------------------------------------\n",
                )
            condition, text = DecisionMakerPerEvent.output_decision(
                condition, results, frames, MLLM
            )
            return condition, text, stored, score_actual
        else:
            return False, "", stored, score_actual

    def process_detections(self, loaded_data, verbose=False):
        persons = DecisionMakerPerEvent.organize_persons(loaded_data)
        stored = []
        score_actual=0
        # Evaluate the persons
        if len(persons) < 2:
            return False, stored,score_actual
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
                return False, stored, score_actual
            for duo in check:
                shaking = DecisionMakerPerEvent.verify_shaking(duo)
                if len(shaking) > 0:
                    stored.append(duo[0])
                    stored.append(duo[1])
                    iou_max = 0
                    for i in range(len(duo[0])):
                        iou = DecisionMakerPerEvent.calculate_shared_area(
                            duo[0][-1], duo[1][-1])
                        if iou > iou_max:
                            iou_max = iou
                    rute_stored='/home/ubuntu/Tesis'
                    file='IOUS_PickPockering.npy'
                    score=duo[0][-1][1] * duo[1][-1][1]*iou_max
                    try:
                        ious=np.load(f"{rute_stored}/{file}", allow_pickle=True)
                    except:
                        ious = np.empty((0, 3), dtype=np.float32) 
                    new_row = np.array([[0, iou_max, duo[0][-1][1] * duo[1][-1][1]]], dtype=ious.dtype)
                    ious = np.vstack((ious, new_row))
                    np.save(f"{rute_stored}/{file}", ious)
                    score_actual=max(score, score_actual)
                    return True, stored, score_actual
        return False, stored, score_actual


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
        scores = []
        print(len(self.__descriptions), len(self.__rules))
        for i in range(len(self.__rules)):
            condition, text, objects , score= self.__rules[i].decision_maker(
                classes, detections, results, frames, MLLM
            )
            prompts.append(self.__descriptions[i])
            scores.append(score)
        #to_recort = self.area_torecort(rois)
        return prompts, scores


if __name__ == "__main__":
    pass
