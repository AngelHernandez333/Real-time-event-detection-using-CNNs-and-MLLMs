import cv2
import time
import os
from MLLMs import LLaVA_OneVision
from Detectors import YOLOv10Detector
import pandas as pd
from Functions3 import take_frame, prompt_text, decision_makerComplex, detection_labels
from Test6Llava import check_precision


def testing(video_path, event, detector_usage, file):
    # Clases de detección
    # Contador de las detecciones
    classes = dict()
    # Inicializadas en cero
    for i in detection_labels.values():
        classes[i] = 0
    # Inicializacion de los modelos
    start_time = time.time()
    # ov_qmodel = YOLOv10("/home/ubuntu/yolov10/int8/yolov10x_openvino_model/")
    if detector_usage != 1:
        ov_qmodel = YOLOv10Detector()
        ov_qmodel.set_model("/home/ubuntu/yolov10/yolov10x.pt")
        ov_qmodel.set_labels(detection_labels)
    if detector_usage < 4:
        llava = LLaVA_OneVision()
        llava.set_model("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
        llava.set_processor("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    cap = cv2.VideoCapture(video_path)
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
    # Charge time
    elapsed_time = time.time() - start_time
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
        if detector_usage == 0:
            # Detector with rules and MLLM with all information
            detections, classes = ov_qmodel.detection(frame, classes)
            decision_makerComplex(
                frame,
                int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                frames,
                5,
                classes,
                detections,
                event,
                results,
            )
        elif detector_usage == 1:
            # Only MLLM
            detections = []
            take_frame(
                frame,
                int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                frames,
                5,
                detections,
                results,
            )
        elif detector_usage == 2:
            # Detector with MLLM but not rules
            detections, classes = ov_qmodel.detection(frame, classes)
            take_frame(
                frame,
                int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                frames,
                5,
                detections,
                results,
            )
        elif detector_usage == 3:
            # Detector with rules and MLLM with no information
            detections, classes = ov_qmodel.detection(frame, classes)
            decision_makerComplex(
                frame,
                int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                frames,
                5,
                classes,
                detections,
                event,
                results,
            )
        elif detector_usage == 4:
            # Detector with rules only
            detections, classes = ov_qmodel.detection(frame, classes)
            decision_makerComplex(
                frame,
                int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                frames,
                5,
                classes,
                detections,
                event,
                results,
                False,
                frames_number,
                prompts,
                file,
            )
        if len(frames) > 6 and detector_usage < 4:
            frames.pop(0)
            results.pop(0)
            frames_number.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            text = prompt_text(classes, event, detector_usage)
            print(text)
            prompt = llava.event_validation(frames, event, text, verbose=True)
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
        cv2.imshow("Video", frame)
    cap.release()
    time_video = time.time() - start_video
    cv2.destroyAllWindows()
    print("Charging time:", elapsed_time, "sg \n\n")
    return frames_number, fps_list, prompts, duration, time_video, finished


if __name__ == "__main__":
    # Videos a usar, descripciones y eventos
    # events=['1-Riding a bicycle', '2-Fight', '3-Playing', '4-Running away', '5-Person lying in the floor',
    #    '6-Chasing', '7-Normal']
    events = [
        "1-Riding a bicycle",
        "2-Fight",
        "3-Playing",
        "4-Running away",
        "5-Person lying in the floor",
        "6-Chasing",
        "7-Normal",
    ]
    description = [
        "a person riding a bicycle",
        "a certain number of persons fighting",
        "a group of persons playing",
        "a person running",
        "a person lying in the floor",
        "a person chasing other person",
        "everything is normal",
    ]
    # Guardar la informacion
    try:
        df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsOOP.csv")
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
    print(df)
    for k in [4]:
        for video_kind in range(len(events)):  # Pasar por todas las carpetas con videos
            rute = f"../Database/CHAD DATABASE/{events[video_kind]}/"
            files = os.listdir(rute)
            for j in range(len(files)):  # Pasar por todos los videos de la carpeta
                # for i in range(len(description)): #Pasar por todas las descripciones
                for i in range(len(description) - 1):
                    count = df[
                        (df["Name"] == files[j])
                        & (df["Check event"] == description[i])
                        & (df["Mode"] == k)
                    ].shape[0]
                    finished = False
                    if count == 0:
                        (
                            frames_number,
                            fps_list,
                            prompts,
                            duration,
                            time_video,
                            finished,
                        ) = testing(
                            f"../Database/CHAD DATABASE/{events[video_kind]}/{files[j]}",
                            description[i],
                            k,
                            files[j],
                        )
                        if finished:
                            frames_number = frames_number[1::]
                            prompts = prompts[1::]
                            print("Prompts:", prompts)
                            tp, fp, fn, tn = check_precision(
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
                            # Append the row to the DataFrame
                            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                            print("\n", df)
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

    df.to_csv("Results/resultsOOP.csv", index=False)
    print("\n", df)
