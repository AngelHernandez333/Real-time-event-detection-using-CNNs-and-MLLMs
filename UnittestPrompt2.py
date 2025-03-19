import unittest
from Functions3 import classes_focus, detection_labels


class MiClaseDePrueba(unittest.TestCase):
    # Métodos de prueba van aquí
    def test_detector_usage(self):
        self.assertEqual(
            prompt_text(
                {"person": 0},
                "a person riding a bicycle",
                3,
                classes_focus,
                [
                    ["car", 0.9324444532394409, 566, 132, 934, 278],
                    ["person", 0.8400475382804871, 391, 69, 429, 163],
                    ["bicycle", 0.8365236520767212, 1001, 154, 1055, 226],
                    ["person", 0.8350211977958679, 251, 131, 318, 211],
                    ["car", 0.8023963570594788, 663, 105, 910, 185],
                    ["person", 0.7844604849815369, 1004, 103, 1054, 213],
                    ["chair", 0.6868844628334045, 70, 251, 186, 420],
                    ["umbrella", 0.6817417144775391, 480, 24, 511, 109],
                    ["umbrella", 0.6096811890602112, 600, 25, 630, 105],
                    ["chair", 0.5393827557563782, 0, 313, 62, 446],
                    ["car", 0.4977438151836395, 710, 53, 758, 92],
                    ["person", 0.47695043683052063, 358, 63, 387, 152],
                    ["dining table", 0.4700835049152374, 0, 288, 91, 315],
                    ["car", 0.3496088683605194, 1267, 132, 1279, 169],
                    ["dining table", 0.2887365221977234, 1, 287, 90, 379],
                    ["person", 0.27817270159721375, 437, 79, 472, 132],
                ],
                (1080, 1920),
            ),
            "Watch the video,",
        )

    def test_stage_inparts(self):
        self.assertEqual(
            prompt_text(
                {"person": 1, "bicycle": 1},
                "a person riding a bicycle",
                0,
                classes_focus,
                [
                    ["car", 0.9324444532394409, 566, 132, 934, 278],
                    ["person", 0.8400475382804871, 391, 69, 429, 163],
                    ["bicycle", 0.8365236520767212, 1001, 154, 1055, 226],
                    ["person", 0.8350211977958679, 251, 131, 318, 211],
                    ["car", 0.8023963570594788, 663, 105, 910, 185],
                    ["person", 0.7844604849815369, 1004, 103, 1054, 213],
                    ["chair", 0.6868844628334045, 70, 251, 186, 420],
                    ["umbrella", 0.6817417144775391, 480, 24, 511, 109],
                    ["umbrella", 0.6096811890602112, 600, 25, 630, 105],
                    ["chair", 0.5393827557563782, 0, 313, 62, 446],
                    ["car", 0.4977438151836395, 710, 53, 758, 92],
                    ["person", 0.47695043683052063, 358, 63, 387, 152],
                    ["dining table", 0.4700835049152374, 0, 288, 91, 315],
                    ["car", 0.3496088683605194, 1267, 132, 1279, 169],
                    ["dining table", 0.2887365221977234, 1, 287, 90, 379],
                    ["person", 0.27817270159721375, 437, 79, 472, 132],
                ],
                (1080, 1920),
            ),
            "The video is divided into nine quadrants, arranged in a 3x3 grid. The quadrants are labeled as Top-Left, Top-Center, Top-Right, Middle-Left, Middle-Center, Middle-Right, Bottom-Left, Bottom-Center, and Bottom-Right. Here is a description of the objects detected in each quadrant: There are 1 person in the Top-Left, 1 person in the Top-Center, and 1 bicycle and 1 person in the Top-Right.",
        )

    def test_nothing(self):
        self.assertEqual(
            prompt_text(
                {"person": 0},
                "a person riding a bicycle",
                0,
                classes_focus,
                [],
                (1080, 1920),
            ),
            "There are no objects of interest in the video.",
        )


def prompt_text(
    classes, event, detector_usage, classes_focus, detections, video_information
):
    if detector_usage > 2:
        return "Watch the video,"
    corrects = []
    for detection in detections:
        if detection[0] in classes_focus[event] and detection[1] > 0.7:
            corrects.append(detection)
    sections = {
        "Top-Left": [],
        "Top-Center": [],
        "Top-Right": [],
        "Middle-Left": [],
        "Middle-Center": [],
        "Middle-Right": [],
        "Bottom-Left": [],
        "Bottom-Center": [],
        "Bottom-Right": [],
    }

    # Example function to determine which section an object is in
    def get_section(x, y, frame_width, frame_height):
        # Calculate the boundaries for each section
        x_step = frame_width // 3
        y_step = frame_height // 3

        if x < x_step:
            x_pos = "Left"
        elif x < 2 * x_step:
            x_pos = "Center"
        else:
            x_pos = "Right"

        if y < y_step:
            y_pos = "Top"
        elif y < 2 * y_step:
            y_pos = "Middle"
        else:
            y_pos = "Bottom"

        return f"{y_pos}-{x_pos}"

    def generate_prompt_with_context(sections):
        # Check if all sections are empty
        if all(not objects for objects in sections.values()):
            return "There are no objects of interest in the video."

        # Introductory description for the MLLM
        intro = (
            "The video is divided into nine quadrants, arranged in a 3x3 grid. "
            "The quadrants are labeled as Top-Left, Top-Center, Top-Right, "
            "Middle-Left, Middle-Center, Middle-Right, Bottom-Left, Bottom-Center, "
            "and Bottom-Right. Here is a description of the objects detected in each quadrant: "
        )

        prompt_parts = []

        for section, objects in sections.items():
            if not objects:
                continue  # Skip empty sections

            # Count the occurrences of each object
            from collections import Counter

            object_counts = Counter(objects)

            # Create a list of descriptions for the objects in this section
            object_descriptions = []
            for obj, count in object_counts.items():
                if count == 1:
                    object_descriptions.append(f"1 {obj}")
                else:
                    object_descriptions.append(f"{count} {obj}s")

            # Join the object descriptions with "and" if there are multiple
            if len(object_descriptions) > 1:
                objects_str = (
                    ", ".join(object_descriptions[:-1])
                    + f" and {object_descriptions[-1]}"
                )
            else:
                objects_str = object_descriptions[0]

            # Add the section description to the prompt parts
            prompt_parts.append(f"{objects_str} in the {section}")

        # Join all section descriptions with commas and "and" for the final prompt
        if len(prompt_parts) > 1:
            objects_description = (
                ", ".join(prompt_parts[:-1]) + f", and {prompt_parts[-1]}"
            )
        else:
            objects_description = prompt_parts[0]

        # Combine the intro and the objects description
        full_prompt = intro + "There are " + objects_description + "."

        return full_prompt

    for detection in corrects:
        section = get_section(
            (detection[4] + detection[2]) / 2,
            (detection[5] + detection[3]) / 2,
            video_information[0],
            video_information[1],
        )
        sections[section].append(detection[0])
    prompt = generate_prompt_with_context(sections)
    return prompt


if __name__ == "__main__":
    unittest.main()
