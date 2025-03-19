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
lists = ["TestingIsThereLlava.csv", "TestingIsThereJanus.csv", "TestingIsThereQwen.csv"]

mAP_models = []
for i in range(len(lists)):
    df = pd.read_csv(f"/home/ubuntu/Tesis/Results/{lists[i]}")
    df = df[df["Mode"] == 1]
    categories = df["True Event"].unique()
    # Separate rows by category
    df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
    df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
    df.fillna(0, inplace=True)
    category_dfs = {
        category: df[df["True Event"] == category] for category in categories
    }

    mAP_process = []
    for i in range(len(categories)):
        df1 = category_dfs[categories[i]]
        #
        df1["Process time"] = df1["Process time"] / df1["Duration"]
        grouped = df1.groupby("Mode")
        # ----------------------------------------------------------------------
        # Ejecución del código
        ap_values = {}
        for mode, group in grouped:
            precision = np.array(group["Precision"].values)
            recall = np.array(group["Recall"].values)
            # Comenta la siguiente línea para verificar si el error es aquí
            ap = calculate_ap(precision, recall)
            ap_values[mode] = ap
        mean_values = grouped[["Precision", "Recall", "Process time"]].mean()
        mean_values["AP"] = [ap_values[mode] for mode in mean_values.index]
        mean_values = mean_values[["AP", "Process time"]]
        mAP_process.append(mean_values)
    # Calculate the mean Average Precision (mAP) for each mode
    mAP_values = pd.concat(mAP_process).groupby(level=0).mean()
    mAP_models.append(mAP_values)
# Combine the list of DataFrames into a single DataFrame
models = ["LLaVa-OneVision-0.5b-ov", "Janus-Pro-1B", "Qwen2-VL-2B-Instruct"]
mAP_combined = pd.concat(mAP_models, keys=models, names=["Model", "Mode"])

# Reset the index to have a flat DataFrame
mAP_combined.reset_index(inplace=True)
mAP_combined.drop(columns="Mode", inplace=True)
print(mAP_combined)


mAP_combined.rename(columns={"AP": "mAP"}, inplace=True)
mAP_combined.rename(columns={"Process time": "Processing time ratio"}, inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

fig.suptitle("Performance evaluation: Only MLLMs", fontsize=16, fontweight="bold")

mAP_combined[["mAP"]].plot(kind="bar", ax=axes[0])
# axes[0].set_title('mAP', fontsize=14, fontweight='bold')
axes[0].set_ylabel("mAP", fontsize=16, fontweight="bold")
axes[0].set_xticklabels(
    mAP_combined["Model"], rotation=0, color="black", fontweight="bold"
)
axes[0].legend().set_visible(False)
axes[0].grid()
axes[0].set_ylim(bottom=0.3)
axes[0].set_xlabel("Configuration", fontsize=16, fontweight="bold").set_visible(False)
axes[0].set_yticklabels(
    ["{:.2f}".format(x) for x in axes[0].get_yticks()], fontsize=10, fontweight="bold"
)

mAP_combined[["Processing time ratio"]].plot(kind="bar", ax=axes[1], color="#ff7f0e")
# axes[1].set_title('Processing time ratio', fontsize=14, fontweight='bold')
axes[1].set_ylabel("Processing time per duration ratio", fontsize=16, fontweight="bold")
axes[1].set_xticklabels(
    mAP_combined["Model"], rotation=0, color="black", fontweight="bold"
)
axes[1].legend().set_visible(False)
axes[1].set_xlabel("Configuration", fontsize=16, fontweight="bold").set_visible(False)
axes[1].grid()
axes[1].set_yticklabels(
    ["{:.2f}".format(x) for x in axes[1].get_yticks()], fontsize=10, fontweight="bold"
)

fig.tight_layout(pad=3.0)
fig.set_size_inches(16, 10)
# plt.savefig("Results/mAP_ProcessingTime.png")
plt.tight_layout()
plt.show()
