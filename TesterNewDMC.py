from abc import ABC, abstractmethod
import cv2
import time
import os
from MLLMs import *
from Detectors import YOLOv10Detector
import pandas as pd
from Functions4 import (
    decision_maker,
    decision_makerComplex,
    classes_focus,
    detection_labels,
)
import numpy as np


class VideoTester(ABC):
    @abstractmethod
    def set_event(self, event):
        pass

    @abstractmethod
    def set_mode(self, mode):
        pass

    @abstractmethod
    def set_rute(self, rute):
        pass

    @abstractmethod
    def set_dataframe(self, df):
        pass

    @abstractmethod
    def set_detector(self, detector):
        pass

    @abstractmethod
    def set_MLLM(self, MLLM):
        pass

    @abstractmethod
    def append_dataframe(self, row):
        pass

    @abstractmethod
    def save_dataframe(self):
        pass

    @abstractmethod
    def show_detections(self):
        pass

    @abstractmethod
    def show_video(self):
        pass

    @abstractmethod
    def check_precision():
        pass

    @abstractmethod
    def testing_video(self):
        pass

    @abstractmethod
    def autotesting(self):
        pass

    @abstractmethod
    def simple_autotesting(self):
        pass

    @abstractmethod
    def set_image_encoder(self):
        pass

    @staticmethod
    def prompt_text(
        classes, event, detector_usage, classes_focus, detections, video_information
    ):
        if detector_usage > 2:
            return "Watch the video."
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
                return "Watch the video."

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

    @staticmethod
    def take_frame(frame, frame_number, frames, gap, detections, results):
        if frame_number % gap == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            results.append(detections)


class EventTester(VideoTester):
    def __init__(self):
        self.__event = None
        self.__mode = None
        self.__rute = None
        self.__df = None
        self.__dfname = None
        self.__showdet = False
        self.__detector = None
        self.__MLLM = None
        self.__image_encoder = None

    def set_detector(self, detector):
        self.__detector = detector

    def set_MLLM(self, MLLM):
        self.__MLLM = MLLM

    def set_event(self, event):
        self.__event = event

    def set_mode(self, mode):
        self.__mode = mode

    def set_rute(self, rute):
        self.__rute = rute

    def show_detections(self, showdet):
        self.__showdet = showdet

    def show_video(self, showvideo):
        self.__showvideo = showvideo

    def set_image_encoder(self, image_encoder):
        self.__image_encoder = image_encoder

    def check_precision(self, prompts, frames_number, video_name):
        # True positive Prediction and Reality are true
        # True negative Prediction and Reality are false
        # False negative Prediction is false and Reality is true
        # False positive Prediction is true and Reality is false
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        prompts = [prompt.lower() for prompt in prompts]
        name = video_name.split(".")[0]
        frames = np.load("..//Database/ALL/GT/gt_ALL.npz")
        frames = frames[name]
        frames = np.append(frames, frames[-1])
        for i in range(len(prompts)):
            print(prompts[i], frames[frames_number[i] - 1], frames_number[i])
            if prompts[i] == "yes" and frames[frames_number[i] - 1] == 1:
                tp += 1
            elif prompts[i] == "yes" and frames[frames_number[i] - 1] == 0:
                fp += 1
            elif prompts[i] == "no" and frames[frames_number[i] - 1] == 1:
                fn += 1
            else:
                tn += 1
        print(tp, fp, fn, tn)
        try:
            return tp, fp, fn, tn
        except:
            return 0, 0, 0, 0

    def set_dataframe(self, df):
        self.__dfname = df
        try:
            df = pd.read_csv(df)
        except:
            columns = [
                "Name",
                "Mode",
                "True Positive",
                "False Positive",
                "False Negative",
                "True Negative",
                "True Event",
                "Check event",
                "Validations Number",
                "Duration",
                "Process time",
            ]
            df = pd.DataFrame(columns=columns)
        self.__df = df
        print("\n", self.__df)

    def append_dataframe(self, row):
        self.__df = pd.concat([self.__df, pd.DataFrame([row])], ignore_index=True)
        print("\n", self.__df)

    def save_dataframe(self):
        print("\n", self.__df)
        self.__df.to_csv(self.__dfname, index=False)

    def testing_video(self, video_path, file):
        # Contador de las detecciones
        classes = dict()
        # Inicializadas en cero
        for i in detection_labels.values():
            classes[i] = 0
        # Inicializacion de los modelos
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_information = (width, height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(video_path)
        print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) ,cap.get(cv2.CAP_PROP_FPS))
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        prev_frame_time = 0
        new_frame_time = 0
        # Almacenamiento de los frames, prompts y resultados
        frames = []
        # Resultados
        frames_number = [0]
        fps_list = []
        prompts = ["Loading..."]
        if self.__mode in [0, 3, 4]:
            dmc = decision_maker(self.__event)
        # Charge time
        prev_frame_time = time.time()
        results = []
        start_video = time.time()
        while True:
            # Leer el siguiente frame
            ret, frame = cap.read()
            if not ret:
                print("No se pudo obtener el frame. Fin del video o error.")
                finished = True
                break
            match self.__mode:
                case 0:
                    # Detector with rules and MLLM with all information
                    detections, classes = self.__detector.detection(frame, classes)
                    decision_makerComplex(
                        frame,
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        frames,
                        5,
                        classes,
                        detections,
                        results,
                        dmc,
                    )
                case 1:
                    # Only MLLM
                    detections = []
                    VideoTester.take_frame(
                        frame,
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        frames,
                        5,
                        detections,
                        results,
                    )
                case 2:
                    # Detector with MLLM but not rules
                    detections, classes = self.__detector.detection(frame, classes)
                    VideoTester.take_frame(
                        frame,
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        frames,
                        5,
                        detections,
                        results,
                    )
                case 3:
                    # Detector with rules and MLLM with no information
                    detections, classes = self.__detector.detection(frame, classes)
                    decision_makerComplex(
                        frame,
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        frames,
                        5,
                        classes,
                        detections,
                        results,
                        dmc,
                    )
                case 4:
                    # Detector with rules only
                    detections, classes = self.__detector.detection(frame, classes)
                    decision_makerComplex(
                        frame,
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        frames,
                        5,
                        classes,
                        detections,
                        results,
                        dmc,
                        False,
                        frames_number,
                        prompts,
                    )
            if len(frames) > 6 and self.__mode < 4:
                frames.pop(0)
                results.pop(0)
                frames_number.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                text = VideoTester.prompt_text(
                    classes,
                    self.__event,
                    self.__mode,
                    classes_focus,
                    detections,
                    video_information,
                )
                prompt = self.__MLLM.event_validation(
                    frames, self.__event, text, verbose=True
                )
                prompts.append(prompt)
            # -------------------------------------
            if cv2.waitKey(1) & 0xFF == ord("q"):
                finished = False
                break
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            time_per_frame = (new_frame_time - prev_frame_time) * 1000
            prev_frame_time = new_frame_time
            print("Los FPS son", fps)
            fps_list.append(fps)
            cv2.putText(
                frame,
                f"Time {time_per_frame:.2f} ms {prompts[-1]}-{len(prompts)-1}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (172, 182, 77),
                2,
            )
            if self.__showdet:
                self.__detector.put_detections(detections, frame)
            if self.__showvideo:
                cv2.imshow("Video", frame)
        cap.release()
        time_video = time.time() - start_video
        cv2.destroyAllWindows()
        return frames_number, fps_list, prompts, duration, time_video, finished

    def autotesting(self, folders, descriptions, modes):
        for k in modes:
            for video_kind in range(len(folders)):
                rute = f"{self.__rute}/{folders[video_kind]}/"
                files = os.listdir(rute)
                for j in range(len(files)):  # Pasar por todos los videos de la carpeta
                    for i in range(len(descriptions)):
                        finished = False
                        if files[j].endswith("_1.mp4"):
                            pass
                        else:
                            continue
                        count = self.__df[
                            (self.__df["Name"] == files[j])
                            & (self.__df["Check event"] == descriptions[i])
                            & (self.__df["Mode"] == k)
                        ].shape[0]
                        if count == 0:
                            self.set_event(descriptions[i])
                            self.set_mode(k)
                            (
                                frames_number,
                                fps_list,
                                prompts,
                                duration,
                                time_video,
                                finished,
                            ) = self.testing_video(
                                f"{self.__rute}/{folders[video_kind]}/{files[j]}",
                                files[j],
                            )
                            if finished:
                                frames_number = frames_number[1::]
                                prompts = prompts[1::]
                                print("Prompts:", prompts)
                                tp, fp, fn, tn = self.check_precision(
                                    prompts, frames_number, files[j]
                                )
                                # Save the results
                                row = {
                                    "Name": files[j],
                                    "Mode": k,
                                    "True Positive": tp,
                                    "False Positive": fp,
                                    "False Negative": fn,
                                    "True Negative": tn,
                                    "True Event": description[video_kind],
                                    "Check event": description[i],
                                    "Validations Number": len(prompts),
                                    "Duration": duration,
                                    "Process time": time_video,
                                }
                                tester.append_dataframe(row)
                                self.save_dataframe()
                            else:
                                break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break

    def simple_autotesting(self, folders, descriptions, modes):
        for k in modes:
            for video_kind in range(len(folders)):
                rute = f"{self.__rute}/{folders[video_kind]}/"
                files = os.listdir(rute)
                for j in range(len(files)):  # Pasar por todos los videos de la carpeta
                    finished = False
                    count = self.__df[
                        (self.__df["Name"] == files[j])
                        & (self.__df["Check event"] == descriptions[video_kind])
                        & (self.__df["Mode"] == k)
                    ].shape[0]
                    check = self.__df[
                        (self.__df["Check event"] == descriptions[video_kind])
                        & (self.__df["Mode"] == k)
                    ].shape[0]
                    if files[j].endswith("_1.mp4"):
                        pass
                    else:
                        continue
                    if count == 0:
                        self.set_event(descriptions[video_kind])
                        self.set_mode(k)
                        (
                            frames_number,
                            fps_list,
                            prompts,
                            duration,
                            time_video,
                            finished,
                        ) = self.testing_video(
                            f"{self.__rute}/{folders[video_kind]}/{files[j]}",
                            files[j],
                        )
                        if finished:
                            frames_number = frames_number[1::]
                            prompts = prompts[1::]
                            print("Prompts:", prompts)
                            tp, fp, fn, tn = self.check_precision(
                                prompts, frames_number, files[j]
                            )
                            # Save the results
                            row = {
                                "Name": files[j],
                                "Mode": k,
                                "True Positive": tp,
                                "False Positive": fp,
                                "False Negative": fn,
                                "True Negative": tn,
                                "True Event": description[video_kind],
                                "Check event": description[video_kind],
                                "Validations Number": len(prompts),
                                "Duration": duration,
                                "Process time": time_video,
                            }
                            tester.append_dataframe(row)
                            self.save_dataframe()
                            # Append the row to the DataFrame
                        else:
                            break
                else:
                    continue
                break
            else:
                continue
            break
        self.save_dataframe()


if __name__ == "__main__":
    # Define the folder of the videos and the descriptions of the events
    # TODO: Verify the events and prompts and test the events

    # Riding a bicycle ✅
    # Fight ✅
    # Playing ✅
    # Running away✅
    # Person lying in the floor✅
    # Chasing✅
    # --------------------Last test were here ------------------------------
    # Jumping ✅
    # Falling ✅
    # Guide✅
    # Littering✅
    # Tripping 🔨Check the prompt
    # Thief 🔨 In process
    # PickPocketing 🔨 In process
    # "a person tripping by other person",
    ov_qmodel = YOLOv10Detector()
    ov_qmodel.set_model("/home/ubuntu/yolov10/yolov10x.pt")
    ov_qmodel.set_labels(detection_labels)

    """events = [
        "1-Riding a bicycle",
        "2-Fight",
        "3-Playing",
        "4-Running away",
        "5-Person lying in the floor",
        "6-Chasing",
    ]
    description = [
        "a person riding a bicycle",
        "a certain number of persons fighting",
        "a group of persons playing",
        "a person running",
        "a person lying in the floor",
        "a person chasing other person",
    ]"""
    """janus = JanusPro()
    janus.set_model("deepseek-ai/Janus-Pro-1B")
    janus.set_processor("deepseek-ai/Janus-Pro-1B")"""
    """events = [
        "1-Riding a bicycle",
        "2-Fight",
        "3-Playing",
        "4-Running away",
        "5-Person lying in the floor",
        "6-Chasing",
        "7-Jumping",
        "8-Falling",
        '9-guide',
        '10-thief',
        '11-Littering',
        "12-Tripping",
        '13-Pickpockering',
    ]"""
    """events = [
        "ALL",
    ]"""
    events = [
        "Riding",
        "Fighting",
        "Playing",
        "Running",
        "Lying",
        "Chasing",
        "Jumping",
        "Falling",
        "Guiding",
        "Stealing",
        "Littering",
        "Tripping",
        "Pickpockering",
    ]
    description = [
        "a person riding a bicycle",
        "a certain number of persons fighting",
        "a group of persons playing",
        "a person running",
        "a person lying in the floor",
        "a person chasing other person",
        "a person jumping",
        "a person falling",
        "a person guiding other person",
        "a person stealing other person",
        "a person throwing trash in the floor",
        "a person tripping",
        "a person stealing other person's pocket",
    ]
    # Prepare the tester
    tester = EventTester()
    test = 1
    if test == 0:
        llava = LLaVA_OneVision()
        llava.set_model("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
        llava.set_processor("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
        tester.set_dataframe("/home/ubuntu/Tesis/Results/TestingDev.csv")
        tester.set_MLLM(llava)
    elif test == 1:
        janus = JanusPro()
        janus.set_model("deepseek-ai/Janus-Pro-1B")
        janus.set_processor("deepseek-ai/Janus-Pro-1B")
        tester.set_dataframe("/home/ubuntu/Tesis/Results/TestingScoreDMC.csv")
        tester.set_MLLM(janus)
    elif test == 2:
        qwen2vl = Qwen2_VL()
        qwen2vl.set_model("Qwen/Qwen2-VL-2B-Instruct")
        qwen2vl.set_processor("Qwen/Qwen2-VL-2B-Instruct")
        tester.set_dataframe("/home/ubuntu/Tesis/Results/TestingIsThereQwen.csv")
        tester.set_MLLM(qwen2vl)
    tester.set_rute("/home/ubuntu/Database/ALL/Videos")
    tester.set_detector(ov_qmodel)
    # tester.set_MLLM(llava)
    tester.show_detections(False)
    tester.show_video(True)
    
    # Start the autotesting
    # tester.autotesting(events, description, [0,1,2,3])
    # tester.simple_autotesting(events, description, [0,1,2,3])
    tester.simple_autotesting(events, description, [4])
    # tester.autotesting(events, description, [0,1,2,3,4])
