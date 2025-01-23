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
df = pd.read_csv("/home/ubuntu/Tesis/Results/results_v2.csv")

# Count the occurrences of each type of event

# Get unique categories
categories = df["Event"].unique()

# Separate rows by category
category_dfs = {category: df[df["Event"] == category] for category in categories}


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
        0: "Detector + Rules + MLLM +Info",
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
# Calculate the mean Average Precision (mAP) for each mode
mAP_values = pd.concat(mAP_process).groupby(level=0).mean()
print(mAP_values)
mAP_values.rename(columns={"AP": "mAP"}, inplace=True)
mAP_values.rename(columns={"Process time": "Processing time ratio"}, inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

mAP_values[["mAP"]].plot(kind="bar", ax=axes[0])
# axes[0].set_title('mAP', fontsize=14, fontweight='bold')
axes[0].set_ylabel("mAP", fontsize=16, fontweight="bold")
axes[0].set_xticklabels(mAP_values.index, rotation=0, color="black", fontweight="bold")
axes[0].legend().set_visible(False)
axes[0].grid()

mAP_values[["Processing time ratio"]].plot(kind="bar", ax=axes[1], color="#ff7f0e")
# axes[1].set_title('Processing time ratio', fontsize=14, fontweight='bold')
axes[1].set_ylabel("Processing time per duration ratio", fontsize=16, fontweight="bold")
axes[1].set_xticklabels(mAP_values.index, rotation=0, color="black", fontweight="bold")
axes[1].legend().set_visible(False)
axes[1].grid()

fig.tight_layout(pad=3.0)
fig.set_size_inches(14, 10)

plt.tight_layout()
plt.show()
plt.show()
