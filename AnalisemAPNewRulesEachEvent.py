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

df = pd.read_csv('/home/ubuntu/Tesis/Results/TestingIsThereLlava.csv')
description = [
    "a person stealing other person",
    "a person throwing trash in the floor",
    "a person tripping",
    "a person stealing other person's pocket",
]

# Filter the DataFrame to only keep rows with True Event in the description list
df = df[df["True Event"].isin(description)]

#  Get unique categories
print(df)
categories = df["True Event"].unique()
print(categories)
# Separate rows by category
df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df.fillna(0, inplace=True)
category_dfs = {category: df[df["True Event"] == category] for category in categories}


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))

fig.suptitle('Events evaluation', fontsize=16, fontweight='bold')
for i in range(len(categories)):
    df1 = category_dfs[categories[i]]
    #
    df1["Process time"] = df1["Process time"] / df1["Duration"]
    print(categories[i])
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
    # Plot the results
    mode_names = {
        0: "Detector + Rules + MLLM + Information",
        1: "MLLM",
        2: "Detector + MLLM + Information ",
        3: "Detector + Rules + MLLM",
        4: "Detector + Rules",
    }
    mean_values.rename(index=mode_names, inplace=True)
    
    mean_values[["AP"]].plot(kind="bar", ax=axes[i//2][i%2])
    axes[i//2][i%2].set_ylabel("AP", fontsize=10, fontweight="bold")
    axes[i//2][i%2].set_xlabel("Mode", fontsize=10, fontweight="bold").set_visible(False)
    axes[i//2][i%2].set_xticklabels(mean_values.index, rotation=45, color="black", fontweight="bold")
    #axes[i//2][i%2].set_yticklabels(np.round(np.linspace(0, 1, 11), 2), color="black", fontweight="bold")
    axes[i//2][i%2].legend().set_visible(False)
    axes[i//2][i%2].grid()
    axes[i//2][i%2].relim()
    axes[i//2][i%2].set_ylim(0, 1)
    axes[i//2][i%2].set_title(f"{categories[i]}", fontsize=14, fontweight="bold")
    '''axes[0].set_ylabel("mAP", fontsize=16, fontweight="bold")
    axes[0].set_xticklabels(mAP_values.index, rotation=0, color="black", fontweight="bold")
    axes[0].legend().set_visible(False)
    axes[0].grid()
    axes[0].set_ylim(bottom=0.0, top=1.0)
    plt.title(f"{categories[i]}")
    plt.xticks(rotation=45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid()
    plt.ylim(bottom=0)'''
    # plt.xlim(0.4)
fig.tight_layout(pad=3.0)
fig.set_size_inches(16, 10)
plt.savefig("Results/Meeting/AP_perevent_Llava.png")
plt.tight_layout()
plt.show()
