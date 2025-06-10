import cv2
from ultralytics import YOLOv10
import math
import time
import numpy as np
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from Functions3 import (
    take_frame,
    prompt_text,
    decision_makerComplex,
    check_precision,
    detection_labels,
)
import pandas as pd
import os


def detection(image, classes, ov_qmodel):
    """
    Funcion de deteccion de objetos en la imagen
    image: imagen a procesar
    """
    detections = []
    results = ov_qmodel(image, stream=False)
    for i in detection_labels.values():
        classes[i] = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Obtener la clase y la confianza
            class_label = int(box.cls[0])  # Convertir a entero si es necesario
            confidence = float(box.conf[0])  # Convertir a flotante si es necesario
            # Muestra la clase y el grado de confianza en el cuadro
            detections.append(
                [detection_labels[class_label], confidence, x1, y1, x2, y2]
            )
            text = f"Class: {detection_labels[class_label] }-{confidence:.2f}, {x1}, {y1}, {x2}, {y2}"
            classes[detection_labels[class_label]] = (
                classes[detection_labels[class_label]] + 1
            )
            """cv2.putText(
                image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 255),
                1,
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)
            confidence = math.ceil((box.conf[0] * 100)) / 100"""
    return detections


# Validacion de la deteccion
def validation(frames, classes, processor, model, prompts, event, detector_usage):
    text = prompt_text(classes, event, detector_usage)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {
                    "type": "text",
                    "text": f"{text} there is {event}? Just yes or no",
                },
            ],
        },
    ]
    video = np.stack(frames)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to(
        "cuda:0", torch.float16
    )
    out = model.generate(**inputs, max_new_tokens=60)
    text_outputs = processor.batch_decode(
        out, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(text_outputs[0].split("\n")[-1])
    prompts.append(text_outputs[0].split("\n")[-1])
    # prompts.append(text_outputs[0])


def testing(video_path, event, end, detector_usage, file):
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
        ov_qmodel = YOLOv10("/home/ubuntu/yolov10/yolov10x.pt")
    if detector_usage < 4:
        # MLLM Llava
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        )
        model.to(dtype=torch.float16, device="cuda")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            detections = detection(frame, classes, ov_qmodel)
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
            detections = detection(frame, classes, ov_qmodel)
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
            detections = detection(frame, classes, ov_qmodel)
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
            detections = detection(frame, classes, ov_qmodel)
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
            validation(
                frames, classes, processor, model, prompts, event, detector_usage
            )
        # -------------------------------------
        if cv2.waitKey(1) & 0xFF == ord("q"):
            finished = False
            break
        if end == 0:
            pass
        elif end <= int(cap.get(cv2.CAP_PROP_POS_FRAMES)):
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
    events = [
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
    ]
    # Guardar la informacion
    columns = [
        "Name",
        "Mode",
        "Precision",
        "Recall",
        "Event",
        "Validations Number",
        "Duration",
        "Process time",
    ]
    # Create an empty DataFrame with the specified columns
    """rute=f"../Database/CHAD DATABASE/{events[5]}/"
    files = os.listdir(rute)
    frames_number, fps_list, prompts, duration, time_video =testing(f"../Database/CHAD DATABASE/{events[5]}/{files[0]}", description[5], 0,4 ) 
    frames_number=frames_number[1::]
    prompts=prompts[1::]
    print('Prompts:', prompts)
    precision, recall = check_precision( prompts, frames_number, files[0])
    print('Precision:', precision, 'Recall:', recall)"""
    # df = pd.read_csv('/home/ubuntu/Tesis/Results/results26Nov.csv')
    # df = pd.read_csv('/home/ubuntu/Tesis/Results/results7v.csv')
    df = pd.DataFrame(columns=columns)
    for video_kind in range(len(events)):
        rute = f"../Database/CHAD DATABASE/{events[video_kind]}/"
        files = os.listdir(rute)
        for j in range(len(files)):
            for k in [1]:
                count = df[
                    (df["Name"] == files[j])
                    & (df["Event"] == description[video_kind])
                    & (df["Mode"] == k)
                ].shape[0]
                if count == 0:
                    frames_number, fps_list, prompts, duration, time_video, finished = (
                        testing(
                            f"../Database/CHAD DATABASE/{events[video_kind]}/{files[j]}",
                            description[video_kind],
                            0,
                            k,
                            files[j],
                        )
                    )
                    if finished:
                        frames_number = frames_number[1::]
                        prompts = prompts[1::]
                        print("Prompts:", prompts)
                        precision, recall = check_precision(
                            prompts, frames_number, files[j]
                        )
                        # Save the results
                        row = {
                            "Name": files[j],
                            "Mode": k,
                            "Precision": precision,
                            "Recall": recall,
                            "Event": description[video_kind],
                            "Validations Number": len(prompts),
                            "Duration": duration,
                            "Process time": time_video,
                        }
                        # Append the row to the DataFrame
                        # df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                        # fps_list = np.array(fps_list)
                        print("\n", df)
                    else:
                        break
            else:
                continue
            break
        else:
            continue
        break

    # df.to_csv('Results/results26Nov2.csv', index=False)
    print("\n", df)
