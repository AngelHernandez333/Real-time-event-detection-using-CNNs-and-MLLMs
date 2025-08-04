import cv2
import time
import os
from MLLMs import *
from Detectors import YOLOv10Detector
import pandas as pd
from Functions4 import (
    detection_labels,classes_focus
)
import numpy as np
from MLLMs import *
from Tester import VideoTester
import pandas as pd
import numpy as np
from CLIPS import CLIP_Model, XCLIP_Model
from DMC_OPP_Score import ALL_Rules

classes_focus = {
    "a person riding a bicycle on the street": ["person", "bicycle"],
    "multiple people engaged in a physical fight": ["person"],
    "a group of people playing a sport together": [
        "person",
        "frisbee",
        "sports ball",
        "baseball glove",
        "tennis racket",
    ],
    "a person running": ["person"],
    "a person lying motionless on the ground": ["person"],
    "a person aggressively chasing another person": ["person"],
    "everything is normal": [
        "person",
        "bicycle",
        "frisbee",
        "sports ball",
        "baseball glove",
        "tennis racket",
    ],
    "a person jumping high in the air with both feet": ["person"],
    "a person accidentally falling to the ground": ["person"],
    "a person gently guiding another person by the arm": ["person"],
    "a person tripping over an obstacle": ["person"],
    "a person deliberately throwing garbage on the ground": ["person"],
    "a person stealing other person": ["person"],
    "a person pickpocketing a wallet from someone's pocket": ["person"],
}


PREFIX = "a video of "


def prompt_text(classes, event, detector_usage, classes_focus):
    if detector_usage > 2:
        return "Watch the video."
    initial = "There are"
    objects = ""
    corrects = []
    for entity in classes_focus[event]:
        if classes[entity] > 0:
            corrects.append(entity)
    if len(corrects) == 1:
        if classes[corrects[0]] == 1:
            objects += f"There is {classes[corrects[0]]} {corrects[0]} in the video."
        else:
            objects += f"There are {classes[corrects[0]]} {corrects[0]}s in the video."
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
        text = "Watch the video."
    else:
        objects = objects[:-1]
        text = f"{initial}{objects} in the video."
    return text


class EventTesterCLIP(VideoTester):
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
        self._storagefolder = "/home/ubuntu/Tesis/Storage/Score_New_Top3"
        self.__order= [
        "Riding",
        "Playing", #Finish specific class events
        "Pickpockering",
        "Stealing",
        "Tripping",
        "Chasing",
        "Guiding",
        "Jumping",
        "Falling",
        "Littering",
        "Running",
        "Lying",
        "Fighting",
    ]
        self.__order_dict = {}

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

    def set_image_encoder(self, image_encoder):
        self.__image_encoder = image_encoder

    def show_detections(self, showdet):
        self.__showdet = showdet

    def show_video(self, showvideo):
        self.__showvideo = showvideo

    def check_precision(
        self,
        frames_number,
        video_name,
        predicted_events,
        event,
        anomaly_classes,
        prompts,
        mode,probabilities
    ):
        # True positive Prediction and Reality are true
        # True negative Prediction and Reality are false
        # False negative Prediction is false and Reality is true
        # False positive Prediction is true and Reality is false
        normal_class = PREFIX + "a normal view (persons walking or standing)"
        all_classes = [normal_class] + anomaly_classes
        name = video_name.split(".")[0]
        frames = np.load("..//Database/ALL/GT/gt_ALL.npz")
        frames = frames[name]
        frames = np.append(frames, frames[-1])
        # Create a dictionary to convert class names to numeric indices
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        # Example: {"a normal view...": 0, "a person riding...": 1, ...}
        num_classes = len(all_classes)  # 14
        cm = np.zeros((num_classes, num_classes), dtype=int)
        # Save frames_number, predicted_events, and prompts into a numpy array
        if self.__mode!=3 or self.__mode!=4:
            prompts = [prompt.lower().split(".")[0] for prompt in prompts]
            output_data = np.array([frames_number, predicted_events, prompts, probabilities], dtype=object)
            np.save(f"{self._storagefolder}/{name}_CLIP_{mode}_{event}.npy", output_data)      
            return 0,0,0,0
            '''for i in range(len(predicted_events)):
                # Get ground truth

                is_anomaly = frames[frames_number[i] - 1]  # 0 or 1

                # Determine true class using EVENT when anomaly exists
                true_class = event if is_anomaly == 1 else normal_class
                print(
                    (frames_number[i] - 1),
                    is_anomaly,
                    true_class,
                    predicted_events[i],
                    prompts[i],
                )

                if prompts[i] == "" and (mode == 0 or mode == 2):
                    pass
                elif prompts[i] == "yes":
                    pass
                elif prompts[i] == "no":
                    continue
                else:
                    continue
                print("Check")
                # Get predicted class
                pred_class = predicted_events[i]

                # Convert to indices (skip if class not recognized)
                true_idx = class_to_idx.get(true_class, -1)
                pred_idx = class_to_idx.get(pred_class, -1)

                if true_idx != -1 and pred_idx != -1:
                    cm[true_idx, pred_idx] += 1
            # Get the index of your event class
            event_idx = class_to_idx[event]

            # Calculate metrics ONLY for your event class
            tp = cm[event_idx, event_idx]  # True positives for event
            fp = (
                np.sum(cm[:, event_idx]) - tp
            )  # False positives (other classes predicted as event)
            fn = (
                np.sum(cm[event_idx, :]) - tp
            )  # False negatives (event misclassified as others)
            tn = np.sum(cm) - tp - fp - fn  # True negatives
            return tp, fp, fn, tn'''
        else:
            output_data = np.array([frames_number, predicted_events, prompts], dtype=object)
            np.save(f"{self._storagefolder}/{name}_CLIP_{mode}_{event}.npy", output_data)
            return 0,0,0,0

    # Rows = true classes, Columns = predicted classes
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

    def autotesting(self, folders, descriptions, modes):
        pass

    def testing_video(self, video_path, dmc):
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
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        prev_frame_time = 0
        new_frame_time = 0
        # Almacenamiento de los frames, prompts y resultados
        frames = []
        # Resultados
        frames_number = [0]
        fps_list = []
        prompts = ["Loading..."]
        events = ["Loading..."]
        probabilities= []
        # Charge time
        prev_frame_time = time.time()
        results = []
        start_video = time.time()
        gap = 5
        padding=[]
        while True:
            # Leer el siguiente frame
            ret, frame = cap.read()
            if not ret:
                print("No se pudo obtener el frame. Fin del video o error.")
                finished = True
                break
            if self.__mode!= 4:
                detections, classes = self.__detector.detection(frame, classes)
            else:
                detections=[]
            VideoTester.take_frame(
                frame,
                int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                frames,
                gap,
                detections,
                results,
            )
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % gap == 0 and self.__mode != 4:
                descriptions, scores = dmc.process(
                        classes, detections, results, frames, False
                    )
            if self.__mode== 4:
                descriptions= dmc.get_descriptions()   
            if len(frames) > 6:
                padding.append(frames[0])
                if len(padding)>2:
                    padding.pop(0)
                frames.pop(0)
                results.pop(0)
                if self.__event in descriptions and self.__mode == 2:
                    descriptions=[self.__event]
                elif self.__mode == 2:
                    descriptions=[]
                #print('Descripciones:',descriptions, '\n')
                normal_prompt = (
                        PREFIX + "a normal view (persons walking or standing)"
                    )
                if len(descriptions) > 0:
                    if normal_prompt not in descriptions:
                        descriptions.append(normal_prompt)
                        scores.append(0)
                    scores= {event: prob for event, prob in zip(descriptions,  scores) if prob > 0}
                    if not scores:
                        prompts.append("")
                    else:
                        # Get the top three scores sorted from max to min
                        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                        top3_events = [event for event, _ in top3]
                        answer=self.__MLLM.event_selection(frames, top3_events, text="Watch the video and", verbose=True)
                        prompts.append(answer)
                    probabilities.append(scores)
                    events.append(descriptions)
                    #print('Descripcion y scores ',descriptions, scores)
                    frames_number.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                else:
                    frames_number.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    events.append(normal_prompt)
                    probabilities.append(0)
                    prompts.append("")
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
                f"Time {time_per_frame:.2f} ms {events[-1]}-{len(events)-1}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (172, 182, 77),
                2,
            )
            if self.__showvideo:
                cv2.imshow("Video", frame)
        cap.release()
        time_video = time.time() - start_video
        cv2.destroyAllWindows()
        return frames_number, fps_list, prompts, duration, time_video, finished, events, probabilities

    def simple_autotesting(self, folders, descriptions, modes):
        dmc = ALL_Rules()
        dmc.set_descriptions(self.__image_encoder.get_descriptions())
        self.__order_dict= {event: PREFIX + desc for event, desc in zip(folders, description)}
        for k in modes:
            for video_kind in range(len(folders)):
                rute = f"{self.__rute}/{folders[video_kind]}/"
                files = os.listdir(rute)
                for j in range(len(files)):  # Pasar por todos los videos de la carpeta
                    finished = False
                    count = self.__df[(self.__df["Mode"] == k)
                        & (self.__df["True Event"] == descriptions[video_kind])
                    ].shape[0]
                    '''if count< 7:
                        pass
                    else:
                        continue'''
                    count = self.__df[
                        (self.__df["Name"] == files[j])
                        & (self.__df["Mode"] == k)
                        & (self.__df["True Event"] == descriptions[video_kind])
                    ].shape[0]
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
                            predicted_events,probabilities
                        ) = self.testing_video(
                            f"{self.__rute}/{folders[video_kind]}/{files[j]}", dmc
                        )
                        if finished:
                            frames_number = frames_number[1::]
                            predicted_events = predicted_events[1::]
                            prompts = prompts[1::]
                            print("Prompts:", prompts)

                            tp, fp, fn, tn = self.check_precision(
                                frames_number,
                                files[j],
                                predicted_events,
                                descriptions[video_kind],
                                descriptions,
                                prompts,
                                k, probabilities
                            )
                            # Save the results
                            row = {
                                "Name": files[j],
                                "Mode": k,
                                "True Positive": tp,
                                "False Positive": fp,
                                "False Negative": fn,
                                "True Negative": tn,
                                "True Event": descriptions[video_kind],
                                "Check event": "",
                                "Validations Number": len(prompts),
                                "Duration": duration,
                                "Process time": time_video,
                            }
                            self.append_dataframe(row)
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
    #First, presence of a specific class
    #Second, groupal events
    #Last, specific events

    description = [
        "a person riding a bicycle on the street",  # Added context
        "multiple people engaged in a physical fight",  # More specific than "fighting"
        "a group of people playing a sport together",  # Added "sport" for visual clarity
        "a person running",  # Added context
        "a person lying motionless on the ground",  # "Motionless" helps distinguish from falling
        "a person aggressively chasing another person",  # "Aggressively" adds distinction
        "a person jumping high in the air with both feet",  # More specific than just "jumping"
        "a person accidentally falling to the ground",  # "Accidentally" helps distinguish
        "a person gently guiding another person by the arm",  # Added detail
        "a person stealing other person",  # More specific than "stealing"
        "a person deliberately throwing garbage on the ground",  # "Deliberately" adds clarity
        "a person tripping over an obstacle",  # More descriptive
        "a person pickpocketing a wallet from someone's pocket",  # Very specific
    ]
    '''
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
    ]'''
    # Prepare the tester
    tester = EventTesterCLIP()
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
        tester.set_dataframe("/home/ubuntu/Tesis/Results/TestingMCNewScoreTOP3.csv")
        tester.set_MLLM(janus)
    elif test == 2:
        qwen2vl = Qwen2_VL()
        qwen2vl.set_model("Qwen/Qwen2-VL-2B-Instruct")
        qwen2vl.set_processor("Qwen/Qwen2-VL-2B-Instruct")
        tester.set_dataframe("/home/ubuntu/Tesis/Results/Testing.csv")
        tester.set_MLLM(qwen2vl)
    '''tester.show_video(False)
    CLIP_encoder = CLIP_Model()
    CLIP_encoder.set_model("openai/clip-vit-base-patch16")
    CLIP_encoder.set_processor("openai/clip-vit-base-patch16")
    descriptions = [PREFIX + des for des in description]
    CLIP_encoder.set_descriptions(descriptions)
    tester.set_image_encoder(CLIP_encoder)'''
    tester.show_video(False)
    CLIP_encoder = XCLIP_Model()
    CLIP_encoder.set_model("microsoft/xclip-base-patch32")
    CLIP_encoder.set_processor("microsoft/xclip-base-patch32")
    descriptions = [PREFIX + des for des in description]
    CLIP_encoder.set_descriptions(descriptions)
    tester.set_image_encoder(CLIP_encoder)
    ov_qmodel = YOLOv10Detector()
    ov_qmodel.set_model("/home/ubuntu/yolov10/yolov10x.pt")
    ov_qmodel.set_labels(detection_labels)
    tester.set_detector(ov_qmodel)
    tester.set_rute("../Database/ALL/Videos")
    tester.show_video(False)
    tester.show_detections(False)
    tester.simple_autotesting(events, descriptions, [0])
