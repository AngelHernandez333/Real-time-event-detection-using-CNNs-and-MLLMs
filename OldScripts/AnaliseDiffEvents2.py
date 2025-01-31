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


df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsLLavaAV_SameVideosDifVal2.csv")
df = df[
    [
        "Name",
        "True Positive",
        "False Positive",
        "False Negative",
        "True Negative",
        "True Event",
        "Check event",
        "Validations Number",
    ]
]
# tp/(tp+fp), tp/(tp+fn)
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df = df[["Name", "Recall", "True Event", "Check event", "Validations Number"]]
df.fillna(0, inplace=True)

df_results = pd.DataFrame(columns=["Name", "True Event", "Predicted event"])
# Check for video
videos = df["Name"].unique()
print(videos)
for i in range(len(videos)):
    video = videos[i]
    df1 = df[df["Name"] == video]
    df1["Softmax Recall"] = softmax(df1["Recall"].values)
    df1 = df1[
        [
            "Name",
            "True Event",
            "Check event",
            "Validations Number",
            "Recall",
            "Softmax Recall",
        ]
    ]
    max_softmax_f1 = df1["Softmax Recall"].max()
    max_check_event = df1[df1["Softmax Recall"] == max_softmax_f1][
        "Check event"
    ].values[0]
    print(df1)
    if df1["Recall"].sum() != 0.0:
        row = {
            "Name": video,
            "True Event": df1["True Event"].unique()[0],
            "Predicted event": max_check_event,
        }
        # Append the row to the DataFrame
        df_results = pd.concat([df_results, pd.DataFrame([row])], ignore_index=True)

df_results["Correct"] = df_results["True Event"] == df_results["Predicted event"]
df_results["Correct"] = df_results["Correct"].astype(int)
print(df_results)
print("Correct precision:", df_results["Correct"].sum() / df_results.shape[0])

events = df_results["True Event"].unique()
print(events)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 12))

# mAP_values[['mAP']].plot(kind='bar', ax=axes[0])

df_compare = pd.DataFrame(columns=["Event", "percentage"])

for i in range(len(events)):
    event = events[i]
    df_event = df_results[df_results["True Event"] == event]
    df_event["Name"] = 1
    print(df_event)
    df_event_grouped = (
        df_event.groupby("Predicted event").size().reset_index(name="Count")
    )
    df_event_grouped["Count"] = (
        df_event_grouped["Count"] / df_event_grouped["Count"].sum()
    )
    print(df_event_grouped)
    colors = [
        "#000055" if x != event else "#D4AF37"
        for x in df_event_grouped["Predicted event"]
    ]
    # Get only the value
    percentage_value = (
        df_event_grouped[df_event_grouped["Predicted event"] == event]["Count"].values[
            0
        ]
        if not df_event_grouped[df_event_grouped["Predicted event"] == event][
            "Count"
        ].empty
        else 0.0
    )
    row = {"Event": event, "percentage": percentage_value}

    # Append the row to the DataFrame
    df_compare = pd.concat([df_compare, pd.DataFrame([row])], ignore_index=True)

    df_event_grouped.plot(
        kind="bar", x="Predicted event", y="Count", ax=axes[i // 3, i % 3], color=colors
    )
    axes[i // 3, i % 3].set_ylabel(
        "Porcentage of predicts", fontsize=12, fontweight="bold"
    )
    axes[i // 3, i % 3].set_xlabel(
        "Predicted event", fontsize=12, fontweight="bold"
    ).set_visible(False)
    axes[i // 3, i % 3].set_xticklabels(
        df_event_grouped["Predicted event"],
        rotation=45,
        color="black",
        fontweight="bold",
        fontsize=9,
    )
    axes[i // 3, i % 3].legend().set_visible(False)
    axes[i // 3, i % 3].set_title(f"{event}", fontsize=12, fontweight="bold")
    axes[i // 3, i % 3].grid()
    # axes[i//3,i%3].set_ylim(bottom=0.5)
fig.tight_layout(pad=3.0)
fig.set_size_inches(14, 10)
# plt.tight_layout()
plt.show()
print(df_compare)
df_compare.to_csv("Results/Analize_Recall.csv", index=False)
