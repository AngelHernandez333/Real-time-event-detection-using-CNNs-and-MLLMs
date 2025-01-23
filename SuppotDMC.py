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
}


def eventsCheck(event, classes, detections, results, frames):
    match event:
        case "a person riding a bicycle":
            # Handle event1
            condition = Check_RidingBicycle(classes, detections)
            return condition
        case "a certain number of persons fighting":
            condition = Check_Fighting(classes, detections)
            return condition
        case "a group of persons playing":
            condition = Check_Playing(classes, detections)
            return condition
        case "a person running":
            condition = Check_Running(classes, detections, results, frames)
            return condition
        case "a person lying in the floor":
            condition = Check_Lying(classes, detections, results, frames)
            return condition
        case _:
            # Handle default case
            return True


def Check_RidingBicycle(classes, detections):
    classes_of_interest = ["person", "bicycle"]
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        return True
    else:
        return False


def Check_Fighting(classes, detections):
    classes_of_interest = ["person"]
    if all([classes[class_] > 1 for class_ in classes_of_interest]):
        correct = []
        for detection in detections:
            if detection[0] == classes_of_interest[0] and detection[1] > 0.5:
                correct.append(detection)
        decision = boxes_touching(correct)
        return decision
    else:
        return False


def Check_Playing(classes, detections):
    classes_of_interest = ["frisbee", "sports ball", "baseball glove", "tennis racket"]
    if (
        any([classes[class_] > 0 for class_ in classes_of_interest])
        and classes["person"] > 1
    ):
        return True
    else:
        return False


def Check_Lying(classes, detections, results, frames):
    classes_of_interest = ["person"]
    print(
        len(results),
        "-",
        len(frames),
        "\n---------------------------------------------------------------------\n",
    )
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True
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
        if condition == False:
            frames.pop(0)
            results.pop(0)
        return condition
    else:
        return False


def Check_Running(classes, detections, results, frames):
    classes_of_interest = ["person"]
    print(
        len(results),
        "-",
        len(frames),
        "\n---------------------------------------------------------------------\n",
    )
    if all([classes[class_] > 0 for class_ in classes_of_interest]):
        if len(results) < 6:
            return True
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
        if condition == False:
            frames.pop(0)
            results.pop(0)
        return condition
    else:
        return False


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


def person_lying(loaded_data):
    # print('Informacion cargada:',loaded_data,'\n---------------------------------------------------------------------\n')
    for i in range(len(loaded_data[0])):
        print(
            "To evalute:",
            loaded_data[0][i],
            "\n---------------------------------------------------------------------\n",
        )
        contador = 0
        for j in range(1, len(loaded_data)):
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
                if calculate_shared_area(loaded_data[0][i], loaded_data[j][k]) > 0.9:
                    # print(j, '-',k,'-',loaded_data[0][i],loaded_data[j][k], calculate_shared_area(loaded_data[0][i], loaded_data[j][k]),'\n')
                    contador += 1
                    # print('Success')
                    break
        print(contador, "----")
        if contador > 4:
            return True
    return False


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
