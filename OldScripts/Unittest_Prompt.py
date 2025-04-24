from Functions3 import classes_focus, detection_labels
import unittest


class MiClaseDePrueba(unittest.TestCase):
    # Métodos de prueba van aquí
    def test_detector_usage(self):
        self.assertEqual(
            prompt_text({"person": 0}, "a person riding a bicycle", 3, classes_focus),
            "Watch the video,",
        )
        self.assertEqual(
            prompt_text({"person": 0}, "a person riding a bicycle", 4, classes_focus),
            "Watch the video,",
        )

    def test_one_entity(self):
        self.assertEqual(
            prompt_text({"person": 1}, "a person riding a bicycle", 0, classes_focus),
            "There is 1 person in the video,",
        )
        self.assertEqual(
            prompt_text({"bicycle": 1}, "a person riding a bicycle", 0, classes_focus),
            "There is 1 bicycle in the video,",
        )

    def test_two_entity(self):
        self.assertEqual(
            prompt_text({"person": 2}, "a person riding a bicycle", 0, classes_focus),
            "There are 2 persons in the video,",
        )
        self.assertEqual(
            prompt_text({"bicycle": 2}, "a person riding a bicycle", 0, classes_focus),
            "There are 2 bicycles in the video,",
        )

    def test_two_entities_two_classes(self):
        self.assertEqual(
            prompt_text(
                {"person": 2, "bicycle": 1},
                "a person riding a bicycle",
                0,
                classes_focus,
            ),
            "There are 2 persons and 1 bicycle in the video,",
        )
        self.assertEqual(
            prompt_text(
                {"bicycle": 2, "person": 1},
                "a person riding a bicycle",
                0,
                classes_focus,
            ),
            "There are 2 bicycles and 1 person in the video,",
        )

    def test_two_entities_three_classes(self):
        self.assertEqual(
            prompt_text(
                {"person": 2, "frisbee": 1, "tennis racket": 1},
                "a group of persons playing",
                0,
                classes_focus,
            ),
            "There are 2 persons, 1 frisbee and 1 tennis racket in the video,",
        )
        self.assertEqual(
            prompt_text(
                {"frisbee": 2, "person": 1, "tennis racket": 2},
                "a group of persons playing",
                0,
                classes_focus,
            ),
            "There are 2 frisbees, 1 person and 2 tennis rackets in the video,",
        )


def prompt_text(classes, event, detector_usage, classes_focus):
    if detector_usage > 2:
        return "Watch the video,"
    initial = "There are"
    objects = ""
    corrects = []
    for entity in classes_focus[event]:
        if classes[entity] > 0:
            corrects.append(entity)
    if len(corrects) == 1:
        if classes[corrects[0]] == 1:
            objects += f"There is {classes[corrects[0]]} {corrects[0]} in the video,"
        else:
            objects += f"There are {classes[corrects[0]]} {corrects[0]}s in the video,"
        return objects
    elif len(corrects) > 1:
        for x in corrects:
            if x == corrects[-1]:
                print(",".join(objects.split(",")[:-1]))
                objects = ",".join(objects.split(",")[:-1])
                objects += f" and"
            objects += f" {classes[x]} {x},"
            if classes[x] > 1:
                objects = objects[:-1] + "s,"
    if objects == "":
        text = "Watch the video,"
    else:
        objects = objects[:-1]
        text = f"{initial}{objects} in the video,"
    return text


if __name__ == "__main__":
    unittest.main()
