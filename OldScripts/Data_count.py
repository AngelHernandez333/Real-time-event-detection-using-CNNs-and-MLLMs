import pandas as pd
import os
import cv2

if __name__ == "__main__":
    rute = "../Database/CHAD DATABASE/"
    events = [
        "1-Riding a bicycle",
        "2-Fight",
        "3-Playing",
        "4-Running away",
        "5-Person lying in the floor",
        "6-Chasing",
        "7-Jumping",
        "8-Falling",
        "9-guide",
        "10-thief",
        "11-Littering",
        "12-Tripping",
        "13-Pickpockering",
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
        "a person littering",
        "a person tripping",
        "a person pickpockering",
    ]
    columns = ["Name", "Event"]
    df = pd.DataFrame(columns=columns)
    for video_kind in range(len(events)):
        actual_rute = f"{rute}/{events[video_kind]}/"
        files = os.listdir(actual_rute)
        print(files)
        for j in range(len(files)):  # Pasar por todos los videos de la carpeta
            cap = cv2.VideoCapture(f"{actual_rute}/{files[j]}")
            duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(
                cv2.CAP_PROP_FPS
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [[files[j], description[video_kind]]], columns=columns
                    ),
                ],
                ignore_index=True,
            )
    # df.to_csv("VideosEventAllDuration.csv", index=False)
    print(df)
    import time

    videos = df["Name"].unique()
    print(len(videos))
    i = 0
    for video in videos:
        df_video = df[df["Name"] == video]
        if df_video.shape[0] > 1:
            combined_events = "; ".join(df_video["Event"].tolist())
            df = df[df["Name"] != video]  # Remove the duplicate rows
            df = pd.concat(
                [df, pd.DataFrame([[video, combined_events]], columns=columns)],
                ignore_index=True,
            )
            print(video, combined_events)
            i += 1
    print(df)
    videos = df["Name"].unique()
    print(len(videos))
    # df.to_csv("VideosEventAll.csv", index=False)
    df.to_csv("VideosEventAllDuration.csv", index=False)
    print("\n\n\n\n\n", len(videos), "Videos unicos", i, "Videos duplicados")
