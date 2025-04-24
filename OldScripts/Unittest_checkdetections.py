import unittest


class TestBase(unittest.TestCase):
    def setUp(self):
        self.classes_of_interest = ["person"]
        self.detections = [
            ["person", 0.8680010437965393, 376, 73, 408, 174],
            ["person", 0.8608394861221313, 1122, 122, 1187, 251],
        ]
        self.results = [
            [
                ["person", 0.8680010437965393, 376, 73, 408, 174],
                ["person", 0.8608394861221313, 1122, 122, 1187, 251],
            ],
            [
                ["person", 0.8680010437965393, 376, 73, 408, 174],
                ["person", 0.8608394861221313, 1122, 122, 1187, 251],
            ],
            [
                ["person", 0.8680010437965393, 376, 73, 408, 174],
                ["person", 0.8608394861221313, 1122, 122, 1187, 251],
            ],
            [
                ["person", 0.8680010437965393, 376, 73, 408, 174],
                ["person", 0.8608394861221313, 1122, 122, 1187, 251],
            ],
            [
                ["person", 0.8680010437965393, 376, 73, 408, 174],
                ["person", 0.8608394861221313, 1122, 122, 1187, 251],
            ],
            [
                ["person", 0.8680010437965393, 376, 73, 408, 174],
                ["person", 0.8608394861221313, 1122, 122, 1187, 251],
            ],
        ]

    def test_lenght(self):
        self.assertEqual(
            len(
                check_detections(
                    self.classes_of_interest, self.detections, self.results
                )
            ),
            7,
        )  # Debería ser 6

    def test_working(self):
        self.assertEqual(
            check_detections(self.classes_of_interest, self.detections, self.results),
            [
                [
                    ["person", 0.8680010437965393, 376, 73, 408, 174],
                    ["person", 0.8608394861221313, 1122, 122, 1187, 251],
                ],
                [
                    ["person", 0.8680010437965393, 376, 73, 408, 174],
                    ["person", 0.8608394861221313, 1122, 122, 1187, 251],
                ],
                [
                    ["person", 0.8680010437965393, 376, 73, 408, 174],
                    ["person", 0.8608394861221313, 1122, 122, 1187, 251],
                ],
                [
                    ["person", 0.8680010437965393, 376, 73, 408, 174],
                    ["person", 0.8608394861221313, 1122, 122, 1187, 251],
                ],
                [
                    ["person", 0.8680010437965393, 376, 73, 408, 174],
                    ["person", 0.8608394861221313, 1122, 122, 1187, 251],
                ],
                [
                    ["person", 0.8680010437965393, 376, 73, 408, 174],
                    ["person", 0.8608394861221313, 1122, 122, 1187, 251],
                ],
                [
                    ["person", 0.8680010437965393, 376, 73, 408, 174],
                    ["person", 0.8608394861221313, 1122, 122, 1187, 251],
                ],
            ],
        )


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


if __name__ == "__main__":
    unittest.main()
