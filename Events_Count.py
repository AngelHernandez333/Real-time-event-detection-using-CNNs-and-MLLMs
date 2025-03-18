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
        "a person throwing trash in the floor",
        'a person tripping',
        "a person stealing other person's pocket",
    ]
    columns = ["Name", "Event"]
    df = pd.DataFrame(columns=columns)
    for video_kind in range(len(events)):
        actual_rute = f"{rute}/{events[video_kind]}/"
        files = os.listdir(actual_rute)
        print(files)
        for j in range(len(files)):  # Pasar por todos los videos de la carpeta
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [[files[j], description[video_kind]]], columns=columns
                    ),
                ],
                ignore_index=True,
            )
    df.to_csv("VideosEventAll.csv", index=False)
    print(df)
    