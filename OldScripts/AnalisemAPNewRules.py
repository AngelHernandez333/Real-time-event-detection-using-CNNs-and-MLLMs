import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_ap(precision, recall):
    # Ordena recall de manera ascendente
    sorted_indices = np.argsort(recall)
    precision = np.array(precision)[sorted_indices]
    recall = np.array(recall)[sorted_indices]

    # Asegúrate de que recall y precision comiencen y terminen en 0 y 1
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))
    # Interpola la precisión para eliminar caídas bruscas
    precision = np.maximum(precision, np.roll(precision, -1))
    # Encuentra puntos donde el recall cambia y calcula el área bajo la curva
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])

    return ap


# Load the CSV file into a DataFrame
# Count the occurrences of each type of event


# df = pd.read_csv("/home/ubuntu/Tesis/Results/Testing6_8.csv")
# df = pd.read_csv("/home/ubuntu/Tesis/Results/RunningLyingOld.csv")
# df=pd.read_csv("/home/ubuntu/Tesis/Results/Results6Events64Videos.csv")
# df=pd.read_csv("/home/ubuntu/Tesis/Results/ResultsNewRules8Events.csv")
# df = pd.read_csv('/home/ubuntu/Tesis/Results/resultsLLavaAV_AllDescriptions.csv')
"""df1 = pd.read_csv("Results/resultsMode1_5Samevideos.csv")
df2 = pd.read_csv("Results/resultsLLavaAV_NormalVideos.csv")
df2["True Event"] = df2["True Event"].replace(
    "a person riding a bicycle", "everything is normal"
)
df3 = pd.read_csv("Results/resultsMode0Samevideos.csv")
df = pd.concat([df1, df2, df3], ignore_index=True)
df = df[df["Check event"] != "everything is normal"]
df = df[df["True Event"] != "everything is normal"]
print(df)
df=df[df['True Event']!= 'everything is normal']
df = df[df['True Event']==df['Check event']]
df= pd.read_csv("Results/TestingDev.csv")"""
df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingIsThereLlava.csv")
df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingJanusAll.csv")
"""description = [
    "a person stealing other person",
    "a person throwing trash in the floor",
    "a person tripping",
    "a person stealing other person's pocket",
]

# Filter the DataFrame to only keep rows with True Event not in the description list
df = df[~df["True Event"].isin(description)]
"""
#  Get unique categories
print(df)
categories = df["True Event"].unique()
print(categories)
# Separate rows by category
df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df.fillna(0, inplace=True)
category_dfs = {category: df[df["True Event"] == category] for category in categories}

mAP_process = []
for i in range(len(categories)):
    df1 = category_dfs[categories[i]]
    #
    df1["Process time"] = df1["Process time"] / df1["Duration"]
    print(df1)
    grouped = df1.groupby("Mode")
    # ----------------------------------------------------------------------
    # Ejecución del código
    ap_values = {}
    for mode, group in grouped:
        precision = np.array(group["Precision"].values)
        recall = np.array(group["Recall"].values)
        print("Mode:", mode, "\n")
        print("Precision:", precision, "Recall:", recall)
        # Comenta la siguiente línea para verificar si el error es aquí
        ap = calculate_ap(precision, recall)
        ap_values[mode] = ap
    mean_values = grouped[["Precision", "Recall", "Process time"]].mean()
    print("Average Precision (AP) for each mode:")
    for mode, ap in ap_values.items():
        print(f"Mode {mode}: {ap:.4f}")
    mean_values["AP"] = [ap_values[mode] for mode in mean_values.index]
    mean_values = mean_values[["AP", "Process time"]]
    mAP_process.append(mean_values)
    # Plot the results
    mode_names = {
        0: "Detector + Rules + MLLM + Information",
        1: "MLLM",
        2: "Detector + MLLM + Information ",
        3: "Detector + Rules + MLLM",
        4: "Detector + Rules",
    }
    mean_values.rename(index=mode_names, inplace=True)
    mean_values[["AP"]].plot(kind="bar")
    plt.title(f"{categories[i]}")
    plt.xlabel("Mode")
    plt.ylabel("Mean Values")
    plt.xticks(rotation=45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid()
    plt.ylim(bottom=0)
    # plt.xlim(0.4)
# Calculate the mean Average Precision (mAP) for each mode
mAP_values = pd.concat(mAP_process).groupby(level=0).mean()
print(mAP_values)
mAP_values.to_csv("/home/ubuntu/Tesis/Results/Meeting/mAPJanus.csv")
mAP_values.rename(columns={"AP": "mAP"}, inplace=True)
mAP_values.rename(columns={"Process time": "Processing time ratio"}, inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

fig.suptitle("Performance evaluation", fontsize=16, fontweight="bold")

mAP_values[["mAP"]].plot(kind="bar", ax=axes[0])
# axes[0].set_title('mAP', fontsize=14, fontweight='bold')
axes[0].set_ylabel("mAP", fontsize=16, fontweight="bold")
axes[0].set_xticklabels(mAP_values.index, rotation=0, color="black", fontweight="bold")
axes[0].legend().set_visible(False)
axes[0].grid()
axes[0].set_ylim(bottom=0.0, top=1.0)
axes[0].set_xlabel("Configuration", fontsize=16, fontweight="bold").set_visible(False)
# axes[0].set_yticklabels(
# ["{:.1f}".format(x) for x in axes[0].get_yticks()], fontsize=10, fontweight="bold"
# )

mAP_values[["Processing time ratio"]].plot(kind="bar", ax=axes[1], color="#ff7f0e")
# axes[1].set_title('Processing time ratio', fontsize=14, fontweight='bold')
axes[1].set_ylabel("Processing time per duration ratio", fontsize=16, fontweight="bold")
axes[1].set_xticklabels(mAP_values.index, rotation=0, color="black", fontweight="bold")
axes[1].legend().set_visible(False)
axes[1].set_xlabel("Configuration", fontsize=16, fontweight="bold").set_visible(False)
axes[1].grid()
axes[1].set_yticklabels(
    ["{:.1f}".format(x) for x in axes[1].get_yticks()], fontsize=10, fontweight="bold"
)

fig.tight_layout(pad=3.0)
fig.set_size_inches(16, 10)
# plt.savefig("Results/Meeting/mAPLLava.png")
plt.tight_layout()
plt.show()
print(df)
print(len(df["Name"].unique()))
print(len(df["True Event"].unique()))
