from Test1Qwen2VLFunc import testing
from Functions3 import check_precision
import pandas as pd
import os

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
    df = pd.DataFrame(columns=columns)
    # df = pd.read_csv('/home/ubuntu/Tesis/Results/results7v.csv')
    video_kind = 0
    rute = f"../Database/CHAD DATABASE/{events[video_kind]}/"
    files = os.listdir(rute)
    for j in range(len(files)):
        for k in range(5):
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
                    print("Precision:", precision, "Recall:", recall)
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
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    # fps_list = np.array(fps_list)
                    print("\n", df)
                else:
                    break
        else:
            continue
        break
    df.to_csv("Results/resultsQwen2VL_6dic.csv", index=False)
    print("\n", df)
