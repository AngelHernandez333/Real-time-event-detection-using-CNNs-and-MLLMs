import numpy as np
import pandas as pd


def eventsCheck(event, classes, detections, results, frames, MLLM):
    match event:
        case "a person lying in the floor":
            condition, prompt = Check_Lying(classes, detections, results, frames, MLLM)
        case "a person chasing other person":
            condition, prompt = Check_Chasing(
                classes, detections, results, frames, MLLM
            )
        case "a person jumping":
            condition, prompt = Check_Jumping(
                classes, detections, results, frames, MLLM
            )
        case "a person falling":
            condition, prompt = Check_Falling(
                classes, detections, results, frames, MLLM
            )
        case "a person guiding other person":
            condition, prompt = Check_Guiding(
                classes, detections, results, frames, MLLM
            )
        case "a person discarding garbage":
            condition, prompt = Check_Littering(
                classes, detections, results, frames, MLLM
            )
        case _:
            # Handle default case
            return True, ""
    return condition, prompt


def check_detections(classes_of_interest, detections, results):
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
    return corrects


def Check_Littering(classes, detections, results, frames, MLLM, frame_number):
    classes_of_interest = ["person"]
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True, ""
        corrects = check_detections(classes_of_interest, detections, results)
        print("Correcs:", corrects)
        condition = person_littering(corrects)
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


def person_littering(loaded_data, verbose=False):
    persons = organize_persons(loaded_data)
    # Evaluate the persons
    if len(persons) < 1:
        return False
    else:
        for person in persons:
            print("Lenght-", len(person))
        return True
    return False


def Check_Guiding(classes, detections, results, frames, MLLM, frame_number):
    classes_of_interest = ["person"]
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True, ""
        corrects = check_detections(classes_of_interest, detections, results)
        print("Correcs:", len(corrects))
        # Save the list to a file
        condition = person_guiding(corrects)
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


def person_guiding(loaded_data, verbose=False):
    persons = organize_persons(loaded_data)
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
                    touch = boxes_touching([persons[i][k], persons[j][k]])
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
                distance1x, distance1y = distance_direction(duo[0][i], duo[0][i + 1])
                distance2x, distance2y = distance_direction(duo[1][i], duo[1][i + 1])
                distancex = np.append(distancex, abs(distance1x - distance2x))
                distancey = np.append(distancey, abs(distance1y - distance2y))
                print(i, "-", distance1x, distance1y, distance2x, distance2y)
            print(distancey.sum(), distancex.sum())
            print(min(widht, height))
            if distancey.sum() < height * 1 and distancex.sum() < widht * 1:
                return True
    return False


def Check_Falling(classes, detections, results, frames, MLLM):
    classes_of_interest = ["person"]
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True, ""
        corrects = check_detections(classes_of_interest, detections, results)
        condition = person_falling(corrects)
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


def person_falling(loaded_data):
    persons = organize_persons(loaded_data)
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


def Check_Jumping(classes, detections, results, frames, MLLM):
    classes_of_interest = ["person"]
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True, ""
        corrects = check_detections(classes_of_interest, detections, results)
        condition = person_jumping(corrects)
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


def person_jumping(loaded_data):
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
        corrects = check_detections(classes_of_interest, detections, results)
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
        corrects = check_detections(classes_of_interest, detections, results)
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
