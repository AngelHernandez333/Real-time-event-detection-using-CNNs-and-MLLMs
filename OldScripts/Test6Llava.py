from Test4Llava4 import testing

# from Functions3 import check_precision
import pandas as pd
import os
import numpy as np


def check_precision(prompts, frames_number, video_name):
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
    frames = np.load(f"../Database/CHAD DATABASE/CHAD_Meta/anomaly_labels/{name}.npy")
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


if __name__ == "__main__":
    # Videos a usar, descripciones y eventos
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
    ]
    # Guardar la informacion
    try:
        df = pd.read_csv(
            "/home/ubuntu/Tesis/Results/resultsLLavaAV_SameVideosDifVal2.csv"
        )
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
    # df = pd.read_csv('/home/ubuntu/Tesis/Results/results7v.csv')
    k = 0  # Mode
    print(df)
    for video_kind in range(len(events)):  # Pasar por todas las carpetas con videos
        rute = f"../Database/CHAD DATABASE/{events[video_kind]}/"
        files = os.listdir(rute)
        for j in range(len(files)):  # Pasar por todos los videos de la carpeta
            for i in range(len(description)):  # Pasar por todas las descripciones
                count = df[
                    (df["Name"] == files[j])
                    & (df["Check event"] == description[i])
                    & (df["Mode"] == k)
                ].shape[0]
                finished = False
                if count == 0:
                    frames_number, fps_list, prompts, duration, time_video, finished = (
                        testing(
                            f"../Database/CHAD DATABASE/{events[video_kind]}/{files[j]}",
                            description[i],
                            0,
                            k,
                            files[j],
                        )
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

    df.to_csv("Results/resultsLLavaAV_SameVideosDifVal3.csv", index=False)
    print("\n", df)
