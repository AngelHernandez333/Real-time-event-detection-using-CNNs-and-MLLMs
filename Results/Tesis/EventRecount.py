import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
    "Andar en bicicleta",
    "Pelear",
    "Jugar",
    "Correr",
    "Estar acostado",
    "Perseguir otra persona",
    "Saltar",
    "Caerse",
    "Guiar a otra persona",
    "Robar a otra persona",
    "Tirar basura",
    "Tropezar con otra persona",
    "Carterismo",
]

columns = [
    "Name",
    "Event",
]
df_newvideos = pd.DataFrame(columns=columns)

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
rute = "../Database/NWPU_IITB/Videos"
for video_kind in range(len(events)):
    actual_rute = f"{rute}/{events[video_kind]}/"
    files = os.listdir(actual_rute)
    for j in range(len(files)):  # Pasar por todos los videos de la carpeta
        df_newvideos = pd.concat(
            [
                df_newvideos,
                pd.DataFrame([[files[j], description[video_kind]]], columns=columns),
            ],
            ignore_index=True,
        )
print(df_newvideos)


df = pd.concat([df_newvideos, df_CHAD], ignore_index=True)
df1 = df_CHAD.groupby("Event").size().reset_index(name="Count")
df2 = df_newvideos.groupby("Event").size().reset_index(name="Count")
df3 = df.groupby("Event").size().reset_index(name="Count")

print(df1, df2, df3)
dfs = [df1, df2, df3]
titles = ["CHAD", "IITB, NWPU y Avenue", "completa"]
all_events = set(df1["Event"]).union(df2["Event"]).union(df3["Event"])

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
            fontsize=12,
            fontweight="bold",
            color="black",
        )
    plt.xlabel("Evento").set_visible(False)
    plt.ylabel("Num. de videos", fontsize=12, fontweight="bold")
    number = dfs[j]["Count"].sum()
    plt.title(
        f"Distribucion de videos en la dataset {titles[j]} con {number} videos.",
        fontsize=16,
        fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=12, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.ylim(bottom=0, top=62)
    plt.grid()
    plt.savefig(
        f"/home/ubuntu/Tesis/Results/Tesis/DistribucionOfTheVideos/Distribution_{titles[j]}.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.gca().set_axisbelow(True)d
plt.show()
