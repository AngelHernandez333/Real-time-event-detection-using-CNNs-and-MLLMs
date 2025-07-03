import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df_NWPU = pd.read_csv("/home/ubuntu/Tesis/Selected_VideosNWPU.csv")
df_IITB = pd.read_csv("/home/ubuntu/Tesis/selected_videosIITB.csv")
df_IITB.rename(columns={"True Event": "Event"}, inplace=True)
columns = [
    "Name",
    "Event",
]
df_CHAD = pd.DataFrame(columns=columns)

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

rute = "../Database/CHAD DATABASE/"
for video_kind in range(len(events)):
    actual_rute = f"{rute}/{events[video_kind]}/"
    files = os.listdir(actual_rute)
    for j in range(len(files)):  # Pasar por todos los videos de la carpeta
        df_CHAD = pd.concat(
            [
                df_CHAD,
                pd.DataFrame([[files[j], description[video_kind]]], columns=columns),
            ],
            ignore_index=True,
        )
print(df_CHAD)
df_IITB.loc[df_IITB["Event"] == "Fight", "Event"] = "Fighting"
df_IITB.loc[df_IITB["Event"] == "Ridijng", "Event"] = "Riding"
print("Events", df_IITB["Event"].unique())
df_NWPU.loc[df_NWPU["Event"] == "Trash", "Event"] = "Littering"
print(df_NWPU["Event"].unique())

df = pd.concat([df_IITB, df_NWPU, df_CHAD], ignore_index=True)

df_IITB.to_csv("/home/ubuntu/Tesis/videosIITBtouse.csv", index=False)
df_NWPU.to_csv("/home/ubuntu/Tesis/videosNWPUtouse.csv", index=False)
df1 = df_CHAD.groupby("Event").size().reset_index(name="Count")
df2 = df_IITB.groupby("Event").size().reset_index(name="Count")
df3 = df_NWPU.groupby("Event").size().reset_index(name="Count")
df4 = df.groupby("Event").size().reset_index(name="Count")
print(df1, df2, df3, df4)
dfs = [df1, df2, df3, df4]
titles = ["CHAD", "IITB", "NWPU", "All"]
all_events = (
    set(df1["Event"]).union(df2["Event"]).union(df3["Event"]).union(df4["Event"])
)

for df in dfs:
    for event in all_events:
        if event not in df["Event"].values:
            df.loc[len(df)] = [event, 0]
    df.sort_values("Event", inplace=True)
    df.reset_index(drop=True, inplace=True)

for j in range(len(dfs)):
    plt.figure(figsize=(10, 6))
    plt.bar(dfs[j]["Event"], dfs[j]["Count"])
    for i, bar in enumerate(plt.gca().patches):
        height = bar.get_height()
        plt.gca().text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )
    plt.xlabel("Event").set_visible(False)
    plt.ylabel("Number of videos", fontsize=12, fontweight="bold")
    number = dfs[j]["Count"].sum()
    plt.title(
        f"Event Distribution in {titles[j]} Dataset with {number} videos.",
        fontsize=16,
        fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=10, fontweight="bold")
    plt.yticks(fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.ylim(bottom=0, top=60)
    plt.grid()
plt.show()
