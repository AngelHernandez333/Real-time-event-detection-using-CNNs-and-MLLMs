import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingJanusAllOnlyTrue.csv")

# df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingJanusCLIP.csv")

# df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingCLIP_RULES32_ALLIMAGES.csv")
# df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingCLIP_RULES32_videoMLLM.csv")
# df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingCLIP_RULES16_videoMLLM.csv")
# df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingCLIP_RULES16_MLLMNewPrompts.csv")

# df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingCLIP_RULES16_MLLMNewPromptsFusion.csv")
# df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingCLIP16_NWPUIITB.csv")
# MLLM 0.431
# ONLY CLIP 0.42813

# df = df[df["Mode"] == 0]
# df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingCLIP_RULES32Cropped.csv")

"""def calculate_ap(precision, recall):
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

    return ap"""
import numpy as np

# For the old prompt
status = "ALL"


if status == "ALL":
    rute = "/home/ubuntu/Tesis/Results/Tesis/Best_CLIP/"
    storing_file = "ALLwithRespectedFPSCLIPEnglishwithONLYCLIP.png"
    file = "TestingCLIP16_NWPUIITB.csv"
    df = pd.read_csv(f"{rute}{file}")
    file = "TestingCLIP_RULES16_MLLMNewPrompts.csv"
    df2 = pd.read_csv(f"{rute}{file}")
    # df["Process time"] = df["Process time"] / df["Duration"]
    # df2["Process time"] = df2["Process time"] / df2["Duration"]
    # df['Process time'] = 25.0 /df['Process time']
    # df2['Process time'] = 30.0 /df2['Process time']
    file = "TestingOnlyCLIP.csv"
    df3 = pd.read_csv(f"{rute}{file}")
    df = pd.concat([df, df2, df3], ignore_index=True)
    df["Process time"] = df["Process time"] / df["Duration"]
    df.loc[df["Name"].str.contains("_1.mp4"), "Process time"] = (
        30.0 / df["Process time"]
    )
    df.loc[~df["Name"].str.contains("_1.mp4"), "Process time"] = (
        25.0 / df["Process time"]
    )
    df = df[(df["Mode"] == 1) | (df["Mode"] == 0)]
elif status == "chad":
    rute = "/home/ubuntu/Tesis/Results/Tesis/Best_CLIP/"
    file = "TestingCLIP_RULES16_MLLMNewPrompts.csv"
    df2 = pd.read_csv(f"{rute}{file}")
    storing_file = file.split(".")[0] + "_mAP.png"
    df = df2
    # df["Process time"] = df["Process time"] / df["Duration"]
    # df2["Process time"] = df2["Process time"] / df2["Duration"]
    # df['Process time'] = 25.0 /df['Process time']
    # df2['Process time'] = 30.0 /df2['Process time']
    df["Process time"] = df["Process time"] / df["Duration"]
    df.loc[df["Name"].str.contains("_1.mp4"), "Process time"] = (
        30.0 / df["Process time"]
    )
    df = df[(df["Mode"] == 1) | (df["Mode"] == 0)]
elif status == "else":
    rute = "/home/ubuntu/Tesis/Results/Tesis/Best_CLIP/"
    file = "TestingCLIP16_NWPUIITB.csv"
    storing_file = file.split(".")[0] + "_mAP.png"
    df = pd.read_csv(f"{rute}{file}")
    # df["Process time"] = df["Process time"] / df["Duration"]
    # df2["Process time"] = df2["Process time"] / df2["Duration"]
    # df['Process time'] = 25.0 /df['Process time']
    # df2['Process time'] = 30.0 /df2['Process time']
    df["Process time"] = df["Process time"] / df["Duration"]
    df.loc[~df["Name"].str.contains("_1.mp4"), "Process time"] = (
        25.0 / df["Process time"]
    )
    df = df[(df["Mode"] == 1) | (df["Mode"] == 0)]


def calculate_ap(precision, recall):
    # Sort by recall (ascending)
    sorted_indices = np.argsort(recall)
    precision = np.array(precision)[sorted_indices]
    recall = np.array(recall)[sorted_indices]

    # Pad with (0,0) and (1,0)
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))

    # Compute AP as the area under the raw curve (no interpolation)
    ap = 0.0
    for i in range(1, len(recall)):
        delta_recall = recall[i] - recall[i - 1]
        ap += delta_recall * precision[i]

    return ap
rute='/home/ubuntu/Tesis/Results/'
file = "TestingDevCLIPOneClassOldDescriptions.csv"
df = pd.read_csv(f"{rute}{file}")
df["Process time"] = df["Process time"] / df["Duration"]
df.loc[df["Name"].str.contains("_1.mp4"), "Process time"] = (
    30.0 / df["Process time"]
)
df.loc[~df["Name"].str.contains("_1.mp4"), "Process time"] = (
    25.0 / df["Process time"]
)
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
    print(df1["Process time"])
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
    """mode_names = {
        0: "M6: CLIP & Rules",1: "M7: CLIP, Rules & and MLLM", 2: "Only CLIP"
    }"""
    mode_names = {
        0: "M6: CLIP con reglas",
        1: "M7: CLIP con reglas y MLLM",
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
print("\n\n", mAP_process)
mAP_values = pd.concat(mAP_process).groupby(level=0).mean()
print("\n\n", mAP_values)
# mAP_values.to_csv("/home/ubuntu/Tesis/Results/Meeting/mAPJanus.csv")
mAP_values.rename(columns={"AP": "mAP"}, inplace=True)
mAP_values.rename(columns={"Process time": "Processing time ratio"}, inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

fig.suptitle("Evaluacion de desempeño: Métodos de CLIP", fontsize=16, fontweight="bold")

mAP_values[["mAP"]].plot(kind="bar", ax=axes[0])
for i, bar in enumerate(axes[0].patches):
    height = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black",
    )
# axes[0].set_title('mAP', fontsize=14, fontweight='bold')
axes[0].set_ylabel("mAP", fontsize=16, fontweight="bold")
axes[0].set_xticklabels(mAP_values.index, rotation=0, color="black", fontweight="bold")
axes[0].legend().set_visible(False)
axes[0].grid()
axes[0].set_ylim(bottom=0.0, top=1.0)
axes[0].set_xlabel("Configuration", fontsize=16, fontweight="bold").set_visible(False)
axes[0].set_yticklabels(
    ["{:.1f}".format(x) for x in axes[0].get_yticks()], fontsize=10, fontweight="bold"
)
# axes[0].set_yticklabels(
# ["{:.1f}".format(x) for x in axes[0].get_yticks()], fontsize=10, fontweight="bold"
# )

mAP_values[["Processing time ratio"]].plot(kind="bar", ax=axes[1], color="#ff7f0e")
for i, bar in enumerate(axes[1].patches):
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black",
    )
# axes[1].set_title('Processing time ratio', fontsize=14, fontweight='bold')
axes[1].set_ylabel("FPS", fontsize=16, fontweight="bold")
axes[1].set_xticklabels(mAP_values.index, rotation=0, color="black", fontweight="bold")
axes[1].legend().set_visible(False)
axes[1].set_xlabel("Configuration", fontsize=16, fontweight="bold").set_visible(False)
axes[1].grid()
axes[1].set_yticklabels(
    ["{:.1f}".format(x) for x in axes[1].get_yticks()], fontsize=10, fontweight="bold"
)

fig.tight_layout(pad=3.0)
fig.set_size_inches(16, 10)
# plt.savefig("Results/Meeting/mAP.png")
plt.tight_layout()
#plt.savefig(f"{rute}{storing_file}", dpi=300, bbox_inches="tight")
plt.show()
print(mAP_values)
# mAP_values.to_csv("/home/ubuntu/Tesis/Results/Meeting/mAPCLIP.csv")
"""print(df[df["True Positive"]==0]['True Event'])
print(df[df["True Positive"]>0]['True Event'])
print(df[df["True Positive"]==0])
print(df[df["True Positive"]>0])"""
