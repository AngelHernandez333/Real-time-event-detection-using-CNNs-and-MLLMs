import cv2
from ultralytics import YOLOv10
import math
import time
import numpy as np
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from Functions import prompt_text, take_frame, detection_labels
import pandas as pd

def detection(image, activation, classes, ov_qmodel, prompts):
    """
    Funcion de deteccion de objetos en la imagen
    image: imagen a procesar
    """
    start_time = time.time()
    if activation:
        results = ov_qmodel(image)
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
                text = f"Class: {detection_labels[class_label] }-{confidence:.2f}"
                classes[detection_labels[class_label]] = (
                    classes[detection_labels[class_label]] + 1
                )
                """cv2.putText(
                    image,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                )
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)"""
                confidence = math.ceil((box.conf[0] * 100)) / 100
    elapsed_time = (time.time() - start_time)*1000
    #print(elapsed_time * 1000, " ms\n")
    cv2.putText(
        image,
        f"Time {elapsed_time:.2f} ms {prompts[-1]}-{len(prompts)-1}",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        1,
    )


# Validacion de la deteccion
def validation(frames,classes, processor, model, prompts, event):    
    text = prompt_text(classes)
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
    #prompts.append(text_outputs[0])


def testing(video_path, event ,end,  starting_frame, detector_usage=False):
    # Clases de detección
    # Contador de las detecciones
    classes = dict()
    # Inicializadas en cero
    for i in detection_labels.values():
        classes[i] = 0

    # Inicializacion de los modelos
    start_time = time.time()
    ov_qmodel = YOLOv10("/home/ubuntu/yolov10/int8/yolov10x_openvino_model/")
    # MLLM Llava
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    model.to(dtype=torch.float16, device="cuda")
    cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_MSEC, 370000)
    cap.set(cv2.CAP_PROP_POS_MSEC, starting_frame)
    prev_frame_time = 0
    new_frame_time = 0
    #Almacenamiento de los frames, prompts y resultados
    frames = []
    #Resultados
    frames_number = [starting_frame]
    fps_list = []
    prompts = ["Loading..."]
    #Charge time
    elapsed_time = time.time() - start_time
    while True:
        # Leer el siguiente frame
        ret, frame = cap.read()
        if not ret:
            print("No se pudo obtener el frame. Fin del video o error.")
            break
        #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        detection(frame, detector_usage, classes, ov_qmodel, prompts )
        take_frame(frame, int(cap.get(cv2.CAP_PROP_POS_FRAMES)), frames,20)
        if len(frames) > 6:
            frames.pop(0)
            frames_number.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES))+starting_frame)
            validation(frames, classes, processor, model, prompts, event)
        # -------------------------------------
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if end==0:
            pass
        elif end <= int(cap.get(cv2.CAP_PROP_POS_FRAMES)):
            break
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print("Los FPS son", fps)
        fps_list.append(fps)
    cap.release()
    cv2.destroyAllWindows()
    print("Charging time:", elapsed_time, "sg \n\n")
    return frames_number, fps_list, prompts

def check_precision( prompts, frames_number, event_interval):
    #True positive Prediction and Reality are true
    #True negative Prediction and Reality are false
    #False negative Prediction is false and Reality is true
    #False positive Prediction is true and Reality is false
    tp=0
    fp=0
    fn=0
    prompts = [prompt.lower() for prompt in prompts]
    for i in range(len(prompts)):
        if prompts[i] == 'yes' and frames_number[i] >= event_interval[0] and frames_number[i] <= event_interval[1]:
            tp+=1
        elif prompts[i] == 'yes' and (frames_number[i] < event_interval[0] or frames_number[i] > event_interval[1]):
            fp+=1
        elif prompts[i] == 'no' and frames_number[i] >= event_interval[0] and frames_number[i] <= event_interval[1]:
            fn+= 1
    print(tp, fp, fn)
    try:
        return tp/(tp+fp), tp/(tp+fn)
    except:
        return 0, 0
if __name__ == "__main__":
    columns = ['ID', 'Detector  use','Precision', 'Recall', 'Event']
    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=columns)
    #Information of the videos
    videos = ["../Database/0.mp4", "../Database/1.mp4", "../Database/2.mp4", "../Database/3.mp4", "../Database/4.mp4"]
    events = ['a motocycle accident', 'an arrest', 'an car accident', 'a fight', 'a person running']
    finishings = [1*60*15, 4*60*15,110*30 ,100*30, 0]
    intervals=[[12*15, 35*15], [100*30, 225*30], [6*30, 54*30], [17*30, 53*30], [1*30, 85*30 ]]
    detections = [True, False]
    #Start
    number=0
    for number in range(len(videos)):
        for detector_status in detections:
            frames_number, fps_list, prompts =testing(videos[number], events[number], finishings[number],0 ,detector_status)
            frames_number=frames_number[1::]
            prompts=prompts[1::]
            #print( prompts, frames_number,'\n')
            precision, recall =check_precision( prompts,frames_number ,intervals[number])
            #print("Precision:", precision, "Recall:", recall)
            #Save the results
            row = {
            'ID': number, 'Detector  use': detector_status,'Precision':precision, 'Recall':recall, 'Event':events[number]}
            # Append the row to the DataFrame
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            fps_list = np.array(fps_list)
            np.save(f'{number}_{detector_status}.npy', fps_list)
        print('\n', df)
    # Save the DataFrame to a CSV file
    df.to_csv('results1.csv', index=False)
#Load an dataframe and save again
