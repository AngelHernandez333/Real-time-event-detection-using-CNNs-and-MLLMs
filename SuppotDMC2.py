import numpy as np
import pandas as pd

classes_focus = {
    "a person riding a bicycle": ["person", "bicycle"],
    "a certain number of persons fighting": ["person"],
    "a group of persons playing": [
        "person",
        "frisbee",
        "sports ball",
        "baseball glove",
        "tennis racket",
    ],
    "a person running": ["person"],
    "a person lying in the floor": ["person"],
    "a person chasing other person": ["person"],
    "everything is normal": [
        "person",
        "bicycle",
        "frisbee",
        "sports ball",
        "baseball glove",
        "tennis racket",
    ],
    "a person jumping": ["person"],
    "a person falling": ["person"],
    "a person guiding someone": ["person"],
}


def eventsCheck(event, classes, detections, results, frames, MLLM, frame_number, file):
    match event:
        case "a person riding a bicycle":
            # Handle event1
            condition, prompt = Check_RidingBicycle(classes, detections)
        case "a certain number of persons fighting":
            condition, prompt = Check_Fighting(classes, detections)
        case "a group of persons playing":
            condition, prompt = Check_Playing(classes, detections)
        case "a person running":
            condition, prompt = Check_Running(
                classes, detections, results, frames, MLLM
            )
        case "a person lying in the floor":
            condition, prompt = Check_Lying(classes, detections, results, frames, MLLM)
        case "a person chasing other person":
            condition, prompt = Check_Chasing(
                classes, detections, results, frames, MLLM
            )
        case "a person jumping":
            condition, prompt = Check_Jumping(
                classes, detections, results, frames, MLLM, frame_number
            )
        case "a person falling":
            condition, prompt = Check_Falling(
                classes, detections, results, frames, MLLM, frame_number
            )
        case "a person guiding someone":
            condition, prompt = Check_Guiding(
                classes, detections, results, frames, MLLM, frame_number
            )
        case _:
            # Handle default case
            return True, ""
    return condition, prompt

def Check_Guiding(classes, detections, results, frames, MLLM, frame_number):
    classes_of_interest = ["person"]
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True, ""
        correct = []
        corrects = []
        for detections in results:
            for detection in detections:
                if detection[0] == classes_of_interest[0] and detection[1] > 0.6:
                    correct.append(detection)
            corrects.append(correct)
            correct = []
        for detection in detections:
            if detection[0] == classes_of_interest[0] and detection[1] > 0.6:
                correct.append(detection)
        corrects.append(correct)
        # Save the list to a file
        condition = person_guiding(corrects, frame_number)
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
    

def person_guiding(loaded_data, frame_number, verbose=False):
    persons = organize_persons(loaded_data)
    # Evaluate the persons
    if len(persons) < 2:
        return False
    else:
        check=[]
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
                    touch= boxes_touching([persons[i][k], persons[j][k]])
                    if touch:
                        touching = np.append(touching , 1)
                    else:
                        touching = np.append(touching , 0)
                if touching.sum() > (len(touching)-1):
                    #return True
                    check.append([persons[i], persons[j]])
        for duo in check:
            for i in range(len(duo[0])):
                print('Duo',[duo[0][i][2]-duo[0][i][4],duo[0][i][3]-duo[0][i][5]],
                    'Duo',[duo[1][i][2]-duo[1][i][4],duo[1][i][3]-duo[1][i][5]], '\n')
    return False


def Check_Falling(classes, detections, results, frames, MLLM, frame_number):
    classes_of_interest = ["person"]
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True, ""
        correct = []
        corrects = []
        for detections in results:
            for detection in detections:
                if detection[0] == classes_of_interest[0] and detection[1] > 0.6:
                    correct.append(detection)
            corrects.append(correct)
            correct = []
        for detection in detections:
            if detection[0] == classes_of_interest[0] and detection[1] > 0.6:
                correct.append(detection)
        corrects.append(correct)
        # Save the list to a file
        condition = person_falling(corrects, frame_number)
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


def person_falling(loaded_data, frame_number):
    persons = organize_persons(loaded_data)
    # Evaluate the persons
    if len(persons) < 1:
        return False
    else:
        for person in persons:
            width_per_height_ratio = np.array([])
            diferences = np.array([])
            for i in range(len(person)):
                print(  i,
                    "-",
                    person[i][2],
                    person[i][3],
                    person[i][4],
                    person[i][5],
                    "\n",)
                diferences = np.append(diferences, person[i][3])
                height = person[i][5] - person[i][3]
                width = person[i][4] - person[i][2]
                width_per_height_ratio = np.append(
                    width_per_height_ratio, width / height
                )
            print(                "Differences:",
                diferences,)
            intervals = diferences - diferences[0]
            width_per_height_ratio = width_per_height_ratio - width_per_height_ratio[0]
            print(
                "Intervals:",
                intervals,
                "Width per height ratio:",
                width_per_height_ratio,
                "Sum difference:",
                intervals.sum(),
                'Sum ratio:',width_per_height_ratio.sum(),
                "\n",
            )
            '''annotations = np.load("../Database/CHAD DATABASE/CHAD_Meta/anomaly_labels/4_089_1.npy")
            if annotations[frame_number - 1] == 1:
                try:
                    df = pd.read_csv("/home/ubuntu/Tesis/Results/Falling12video.csv")
                except:
                    columns = ['Frame','Ratio sum',"Y sum"]
                    df = pd.DataFrame(columns=columns)
                row = {'Frame':frame_number,'Ratio sum':abs(width_per_height_ratio.sum()),"Y sum":abs(intervals.sum())}
                df=pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.to_csv("/home/ubuntu/Tesis/Results/Falling12video.csv", index=False)
                print(df)'''
            if (
                abs(intervals.sum()) > 100 and abs(width_per_height_ratio.sum()) > 0.5
            ):
                return True
    return False

def Check_Jumping(classes, detections, results, frames, MLLM, frame_number):
    classes_of_interest = ["person"]
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True, ""
        correct = []
        corrects = []
        for detections in results:
            for detection in detections:
                if detection[0] == classes_of_interest[0] and detection[1] > 0.6:
                    correct.append(detection)
            corrects.append(correct)
            correct = []
        for detection in detections:
            if detection[0] == classes_of_interest[0] and detection[1] > 0.6:
                correct.append(detection)
        corrects.append(correct)
        # Save the list to a file
        condition = person_jumping(corrects, frame_number)
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


def person_jumping(loaded_data, frame_number):
    persons = organize_persons(loaded_data)
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
            """annotations = np.load("../Database/CHAD DATABASE/CHAD_Meta/anomaly_labels/1_092_1.npy")
        if annotations[frame_number - 1] == 1:
            try:
                df = pd.read_csv("/home/ubuntu/Tesis/Results/Jumping2video.csv")
            except:
                columns = ['Frame',"Height","Mean"]
                df = pd.DataFrame(columns=columns)
            row = {'Frame':frame_number,"Height":height,"Mean":abs(intervals.sum())}
            df=pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv("/home/ubuntu/Tesis/Results/Jumping2video.csv", index=False)
            print(df) """
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
                distance, same_person = distance_between(
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


def Check_RidingBicycle(classes, detections):
    classes_of_interest = ["person", "bicycle"]
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        correct = []
        for detection in detections:
            if detection[0] in classes_of_interest and detection[1] > 0.5:
                correct.append(detection)
        decision = boxes_touching(correct)
        if decision:
            text = "yes"
        else:
            text = "no"
        return decision, text
    else:
        return False, ""


def Check_Fighting(classes, detections):
    classes_of_interest = ["person"]
    if all([classes[class_] > 1 for class_ in classes_of_interest]):
        correct = []
        for detection in detections:
            if detection[0] == classes_of_interest[0] and detection[1] > 0.5:
                correct.append(detection)
        decision = boxes_touching(correct)
        if decision:
            text = "yes"
        else:
            text = "no"
        return decision, text
    else:
        return False, ""


def Check_Playing(classes, detections):
    classes_of_interest = ["frisbee", "sports ball", "baseball glove", "tennis racket"]
    if (
        any([classes[class_] > 0 for class_ in classes_of_interest])
        and classes["person"] > 1
    ):
        return True, "yes"
    else:
        return False, "no"


def Check_Lying(classes, detections, results, frames, MLLM):
    classes_of_interest = ["person"]
    print(
        len(results),
        "-",
        len(frames),
        "\n---------------------------------------------------------------------\n",
    )
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True, ""
        correct = []
        corrects = []
        for detections in results:
            for detection in detections:
                if detection[0] == classes_of_interest[0] and detection[1] > 0.7:
                    correct.append(detection)
            corrects.append(correct)
            correct = []
        for detection in detections:
            if detection[0] == classes_of_interest[0] and detection[1] > 0.7:
                correct.append(detection)
        corrects.append(correct)
        # Save the list to a file
        condition = person_lying2(corrects)
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


def Check_Running(classes, detections, results, frames, MLLM):
    classes_of_interest = ["person"]
    print(
        len(results),
        "-",
        len(frames),
        "\n---------------------------------------------------------------------\n",
    )
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True, ""
        correct = []
        corrects = []
        for detections in results:
            for detection in detections:
                if detection[0] == classes_of_interest[0] and detection[1] > 0.7:
                    correct.append(detection)
            corrects.append(correct)
            correct = []
        for detection in detections:
            if detection[0] == classes_of_interest[0] and detection[1] > 0.7:
                correct.append(detection)
        corrects.append(correct)
        # Save the list to a file
        condition = person_running(corrects)
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


def Check_Chasing(classes, detections, results, frames, MLLM):
    classes_of_interest = ["person"]
    print(
        len(results),
        "-",
        len(frames),
        "\n---------------------------------------------------------------------\n",
    )
    if all([classes[class_] > 1 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True, ""
        correct = []
        corrects = []
        for detections in results:
            for detection in detections:
                if detection[0] == classes_of_interest[0] and detection[1] > 0.6:
                    correct.append(detection)
            corrects.append(correct)
            correct = []
        for detection in detections:
            if detection[0] == classes_of_interest[0] and detection[1] > 0.6:
                correct.append(detection)
        corrects.append(correct)
        # Save the list to a file
        condition = person_chasing(corrects)
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


def do_rectangles_touch(rect1, rect2):
    """
    Check if two rectangles touch each other.

    Parameters:
    rect1 (tuple): A tuple (x1, y1, x2, y2) representing the first rectangle.
    rect2 (tuple): A tuple (x1, y1, x2, y2) representing the second rectangle.

    Returns:
    bool: True if the rectangles touch each other, False otherwise.
    """
    x1_1, y1_1, x2_1, y2_1 = rect1[2], rect1[3], rect1[4], rect1[5]
    x1_2, y1_2, x2_2, y2_2 = rect2[2], rect2[3], rect2[4], rect2[5]

    # Check if the rectangles overlap horizontally and vertically
    horizontal_overlap = not (x2_1 < x1_2 or x2_2 < x1_1)
    vertical_overlap = not (y2_1 < y1_2 or y2_2 < y1_1)

    return horizontal_overlap and vertical_overlap


def boxes_touching(correct):
    """
    Check if any pair of rectangles in a list touch each other.

    Parameters:
    rectangles (list): A list of tuples, where each tuple represents a rectangle (x1, y1, x2, y2).

    Returns:
    bool: True if any pair of rectangles touch each other, False otherwise.
    """
    n = len(correct)
    for i in range(n):
        for j in range(i + 1, n):
            if do_rectangles_touch(correct[i], correct[j]):
                return True
    return False


def calculate_shared_area(rect1, rect2):
    """
    Calculate the shared area between two rectangles.

    Parameters:
    rect1 (tuple): A tuple (x1, y1, x2, y2) representing the first rectangle.
    rect2 (tuple): A tuple (x1, y1, x2, y2) representing the second rectangle.

    Returns:
    float: The shared area between the two rectangles.
    """
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


def person_lying2(loaded_data):
    # print('Informacion cargada:',loaded_data,'\n---------------------------------------------------------------------\n')
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
                    calculate_shared_area(loaded_data[0][i], loaded_data[j][k]),
                    "\n",
                )
                if calculate_shared_area(loaded_data[0][i], loaded_data[j][k]) > 0.93:
                    if stored_data == []:
                        stored_data.append(
                            [
                                loaded_data[j][k],
                                calculate_shared_area(
                                    loaded_data[0][i], loaded_data[j][k]
                                ),
                            ]
                        )
                    else:
                        if stored_data[0][1] < calculate_shared_area(
                            loaded_data[0][i], loaded_data[j][k]
                        ):
                            stored_data[0] = [
                                loaded_data[j][k],
                                calculate_shared_area(
                                    loaded_data[0][i], loaded_data[j][k]
                                ),
                            ]
            if len(stored_data) > 0:
                contador += 1
                loaded_data[0][i] = stored_data[0][0]
        print(contador, "----")
        if contador > 5:
            return True
    return False


def person_running(loaded_data):
    # print('Informacion cargada:',loaded_data,'\n---------------------------------------------------------------------\n')
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
                    calculate_shared_area(loaded_data[0][i], loaded_data[j][k]),
                    "\n",
                )
                if (
                    calculate_shared_area(loaded_data[0][i], loaded_data[j][k]) > 0.2
                    and calculate_shared_area(loaded_data[0][i], loaded_data[j][k])
                    < 0.9
                ):
                    if stored_data == []:
                        stored_data.append(
                            [
                                loaded_data[j][k],
                                calculate_shared_area(
                                    loaded_data[0][i], loaded_data[j][k]
                                ),
                            ]
                        )
                    else:
                        if stored_data[0][1] < calculate_shared_area(
                            loaded_data[0][i], loaded_data[j][k]
                        ):
                            stored_data[0] = [
                                loaded_data[j][k],
                                calculate_shared_area(
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


def person_chasing(loaded_data):
    # print('Informacion cargada:',loaded_data[0],'\n---------------------------------------------------------------------\n')
    # Evaluate the detections and find all the persons
    persons = organize_persons(loaded_data)
    # Evaluate the persons
    if len(persons) < 2:
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
                    distance, _ = distance_between(persons[i][k], persons[j][k])
                    diferences = np.append(diferences, distance)
                """intervals=np.diff(diferences)
                print('Diferences:', diferences, 'Intervals:', intervals, 'Mean:', intervals.mean(), '\n')
                if abs(intervals.mean()) < diferences[0]*0.5 and abs(intervals.mean())> diferences[0]*0.01:
                    return True
                else:
                    return False"""
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
    return False


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
