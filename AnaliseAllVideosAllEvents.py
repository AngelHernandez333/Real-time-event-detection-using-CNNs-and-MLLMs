import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("Results/resultsMode1_5Samevideos.csv")
df2 = pd.read_csv("Results/resultsLLavaAV_NormalVideos.csv")
df2["True Event"] = df2["True Event"].replace(
    "a person riding a bicycle", "everything is normal"
)
df3 = pd.read_csv("Results/resultsMode0Samevideos.csv")
df = pd.concat([df1, df2, df3], ignore_index=True)
df = df[df["Check event"] != "everything is normal"]
df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df.fillna(0, inplace=True)
df["F1"] = 2 * (df["Precision"] * df["Recall"]) / (df["Precision"] + df["Recall"])
df.fillna(0, inplace=True)
df.drop(
    columns=[
        "True Positive",
        "False Positive",
        "False Negative",
        "True Negative",
        "Precision",
        "Recall",
    ],
    inplace=True,
)

modes = df["Mode"].unique()
df_results = pd.DataFrame(columns=["Name", "True Event", "Predicted event", "Mode"])
df_time = pd.DataFrame(columns=["Mode", "Duration", "Process Time"])
for j in modes:
    df_mode = df[df["Mode"] == j]
    videos = df_mode["Name"].unique()
    for i in range(len(videos)):
        video = videos[i]
        df1 = df_mode[df_mode["Name"] == video]
        df1 = df1[["Name", "True Event", "Check event", "Validations Number", "F1"]]
        max_f1 = df1["F1"].max()
        max_check_event = df1[df1["F1"] == max_f1]["Check event"].values[0]
        if df1["F1"].sum() != 0.0:
            row = {
                "Name": video,
                "True Event": df1["True Event"].unique()[0],
                "Predicted event": max_check_event,
                "Mode": j,
            }
            # Append the row to the DataFrame
            df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
        else:
            row = {
                "Name": video,
                "True Event": df1["True Event"].unique()[0],
                "Predicted event": "everything is normal",
                "Mode": j,
            }
            # Append the row to the DataFrame
            df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
    row = {
        "Mode": j,
        "Duration": df_mode["Duration"].sum(),
        "Process Time": df_mode["Process time"].sum(),
    }
    # Append the row to the DataFrame
    df_time = pd.concat([df_time, pd.DataFrame([row])], ignore_index=True)
df_time["Duration"] = df_time["Duration"] / (60 * 60)
df_time["Process Time"] = df_time["Process Time"] / (60 * 60)
df_time.set_index("Mode", inplace=True)
df_time["Process Time Ratio"] = df_time["Process Time"] / df_time["Duration"]
mode_names = {
    0: "Detector + Rules + MLLM +Info",
    1: "MLLM",
    2: "Detector + MLLM + Information ",
    3: "Detector + Rules + MLLM",
    4: "Detector + Rules",
}
df_time.rename(index=mode_names, inplace=True)

df_results["Correct"] = df_results["True Event"] == df_results["Predicted event"]
df_results["Correct"] = df_results["Correct"].astype(int)
# print('Correct precision:', df_results['Correct'].sum()/df_results.shape[0])
# print(df_results[df_results['True Event']=='everything is normal'])
modes = df_results["Mode"].unique()
events = df_results["True Event"].unique()
df_compare = pd.DataFrame(columns=["Event", "percentage", "Mode"])

for mode in modes:
    df_mode = df_results[df_results["Mode"] == mode]
    for i in range(len(events)):
        event = events[i]
        df_event = df_mode[df_mode["True Event"] == event]
        df_event["Name"] = 1
        df_event_grouped = (
            df_event.groupby("Predicted event").size().reset_index(name="Count")
        )
        df_event_grouped["Count"] = (
            df_event_grouped["Count"] / df_event_grouped["Count"].sum()
        )
        percentage_value = (
            df_event_grouped[df_event_grouped["Predicted event"] == event][
                "Count"
            ].values[0]
            if not df_event_grouped[df_event_grouped["Predicted event"] == event][
                "Count"
            ].empty
            else 0.0
        )
        row = {"Event": event, "percentage": percentage_value, "Mode": mode}
        # Append the row to the DataFrame
        df_compare = pd.concat([df_compare, pd.DataFrame([row])], ignore_index=True)
df_compare = df_compare.groupby(["Event", "Mode"]).mean()
df_compare = df_compare.groupby(["Mode"]).mean()
# -------------------Plotting
fig, axes = plt.subplots(nrows=1, ncols=2)
df_time = df_time.sort_values(by="Process Time Ratio")

df_time[["Process Time Ratio"]].plot(kind="bar", color="#004565", ax=axes[0])
axes[0].set_xlabel("Mode").set_visible(False)
axes[0].set_ylabel("Processing Ratio", fontsize=12, fontweight="bold")
axes[0].axhline(1, color="#005500", linestyle="--")
axes[0].set_xticklabels(
    df_time.index, rotation=45, ha="right", fontsize=9, color="black", fontweight="bold"
)
axes[0].set_ylim(0.8)
axes[0].grid()
axes[0].legend(loc="best").set_visible(False)
# plt.tight_layout()
mode_names = {
    0: "Detector + Rules + MLLM +Info",
    1: "MLLM",
    2: "Detector + MLLM + Information ",
    3: "Detector + Rules + MLLM",
    4: "Detector + Rules",
}
df_compare.rename(index=mode_names, inplace=True)
df_compare = df_compare.sort_values(by="percentage")
df_compare.plot(kind="bar", color="#006545", ax=axes[1])
axes[1].set_xlabel("Mode").set_visible(False)
axes[1].set_ylabel(r"Correct clasification ($\%$)", fontsize=12, fontweight="bold")
axes[1].set_xticklabels(
    df_compare.index,
    rotation=45,
    ha="right",
    fontsize=9,
    color="black",
    fontweight="bold",
)
axes[1].legend(loc="best").set_visible(False)
axes[1].set_ylim(0.4, 0.65)
axes[1].grid()
plt.tight_layout()

plt.show()
