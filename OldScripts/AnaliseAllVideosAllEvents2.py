import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def softmax(z):
    """
    Compute the softmax of vector z.

    Parameters:
    z (numpy.ndarray): Input vector.

    Returns:
    numpy.ndarray: Softmax of the input vector.
    """
    exp_z = np.exp(z - np.max(z))  # Subtract max(z) for numerical stability
    return exp_z / np.sum(exp_z)


df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingJanusAll.csv")
df = df[df["Mode"] == 0]
df["True Event"] = "To check"
df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df = df[
    ["Name", "Precision", "Recall", "True Event", "Check event", "Validations Number"]
]
df.fillna(0, inplace=True)
df["F1"] = 2 * (df["Precision"] * df["Recall"]) / (df["Precision"] + df["Recall"])
df.fillna(0, inplace=True)

df_results = pd.DataFrame(columns=["Name", "True Event", "Predicted event"])
# Check for video
videos = df["Name"].unique()
for i in range(len(videos)):
    video = videos[i]
    df1 = df[df["Name"] == video]
    df1["Softmax F1"] = softmax(df1["F1"].values)
    df1 = df1[
        ["Name", "True Event", "Check event", "Validations Number", "F1", "Softmax F1"]
    ]
    max_softmax_f1 = df1["Softmax F1"].max()
    max_check_event = df1[df1["Softmax F1"] == max_softmax_f1]["Check event"].values[0]
    print(df1)
    if df1["F1"].sum() != 0.0:
        row = {
            "Name": video,
            "True Event": df1["True Event"].unique()[0],
            "Predicted event": max_check_event,
        }
        # Append the row to the DataFrame
        df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
    else:
        row = {
            "Name": video,
            "True Event": df1["True Event"].unique()[0],
            "Predicted event": "everything is normal",
        }
        # Append the row to the DataFrame
        df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)
df_trueEvents = pd.read_csv("/home/ubuntu/Tesis/VideosEventAll.csv")
df_trueEvents["Predicted Event"] = "To check"
print(df_results)
print(df_trueEvents)
# Iterate through all rows in df_trueEvents
for index, row in df_results.iterrows():
    video_name = row["Name"]
    event = row["Predicted event"]
    # Update the True Event in df_results if the video name matches
    df_trueEvents.loc[df_trueEvents["Name"] == video_name, "Predicted Event"] = event
print(df_trueEvents)

df_trueEvents["Correct"] = df_trueEvents["Predicted Event"] == df_trueEvents["Event"]
df_trueEvents["Correct"] = df_trueEvents["Correct"].astype(int)
df_trueEvents.to_csv("/home/ubuntu/Tesis/PredictedMode0.csv", index=False)
print(df_trueEvents)
print("Correct precision:", df_trueEvents["Correct"].sum() / df_trueEvents.shape[0])

events = df_trueEvents["Event"].unique()
print(events)
df_compare = pd.DataFrame(columns=["Event", "Percentage"])

for i in range(len(events)):
    event = events[i]
    df_event = df_trueEvents[df_trueEvents["Event"] == event]
    df_event["Name"] = 1
    print(df_event)
    df_event_grouped = (
        df_event.groupby("Predicted Event").size().reset_index(name="Count")
    )
    df_event_grouped["Count"] = (
        df_event_grouped["Count"] / df_event_grouped["Count"].sum()
    )
    print(df_event_grouped)
    colors = [
        "#000055" if x != event else "#D4AF37"
        for x in df_event_grouped["Predicted Event"]
    ]

    # Get only the value
    percentage_value = (
        df_event_grouped[df_event_grouped["Predicted Event"] == event]["Count"].values[
            0
        ]
        if not df_event_grouped[df_event_grouped["Predicted Event"] == event][
            "Count"
        ].empty
        else 0.0
    )
    row = {"Event": event, "Percentage": percentage_value}

    # Append the row to the DataFrame
    df_compare = pd.concat([df_compare, pd.DataFrame([row])], ignore_index=True)

    # Create individual plot for each event
    df_event_grouped.plot(kind="bar", x="Predicted Event", y="Count", color=colors)
    plt.ylabel("Correct Predictions (%)", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted event", fontsize=12, fontweight="bold").set_visible(False)
    plt.xticks(rotation=45, fontsize=10, fontweight="bold")
    plt.legend().set_visible(False)
    plt.title(f"{event}", fontsize=12, fontweight="bold")
    plt.grid()
    plt.tight_layout()
    # plt.savefig(f'Results/Analize_F1_{event}.png')
plt.show()
print(df_compare["Percentage"].mean())
