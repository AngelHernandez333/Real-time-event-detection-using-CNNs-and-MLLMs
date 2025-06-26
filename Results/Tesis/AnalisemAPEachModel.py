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
lists = ["TestingIsThereLlava.csv", "TestingJanusIsThere.csv", "TestingIsThereQwen.csv"]

mAP_models = []
for i in range(len(lists)):
    df = pd.read_csv(f"/home/ubuntu/Tesis/Results/Tesis/SelectionOfModel/{lists[i]}")
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
        df1['F1']= 2 * (df1['Precision'] * df1['Recall']) / (df1['Precision'] + df1['Recall'])
        df1.fillna(0, inplace=True)
        df1["Process time"] = df1["Process time"] / df1["Duration"]
        df1['Process time'] = 30.0 /df1['Process time']
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
        mean_values = grouped[["Precision", "Recall", "Process time", 'F1']].mean()
        print(f'Processing category: {categories[i]}',mean_values)
        mean_values["AP"] = [ap_values[mode] for mode in mean_values.index]
        mean_values = mean_values[["AP", "Process time", "F1"]]
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

#mAP_combined.rename(columns={"AP": "mAP"}, inplace=True)
mAP_combined.rename(columns={"Process time": "Processing time ratio"}, inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

fig.suptitle("Evaluacion de desempeño: Método M1: MLLM Solo", fontsize=16, fontweight="bold")

'''mAP_combined[["mAP"]].plot(kind="bar", ax=axes[0])
mAP_combined[["F1"]].plot(kind="bar", ax=axes[0])
for i, bar in enumerate(axes[0].patches):
    height = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
    )'''
# Plot mAP and F1 side by side
width = 0.4  # Width of the bars
x = np.arange(len(mAP_combined))  # X positions for the bars

axes[0].bar(x - width / 2, mAP_combined["AP"], width, label="AP", color="#1f77b4")
axes[0].bar(x + width / 2, mAP_combined["F1"], width, label="F1", color="#2ca02c")

# Add labels above the bars
for i, (map_val, f1_val) in enumerate(zip(mAP_combined["AP"], mAP_combined["F1"])):
    axes[0].text(
        x[i] - width / 2,
        map_val,
        f"{map_val:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
    )
    axes[0].text(
        x[i] + width / 2,
        f1_val,
        f"{f1_val:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
    )

# Configure the axes
axes[0].set_ylabel("Scores", fontsize=16, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(
    mAP_combined["Model"], rotation=0, color="black", fontweight="bold"
)
legend = axes[0].legend(["AP", "F1"], loc="upper right", prop={"weight": "bold"})
#legend.set_title("Legend", prop={"size": 18, "weight": "bold"})
for text in legend.get_texts():
    text.set_fontsize(13)
axes[0].grid()
axes[0].set_ylim(bottom=0.3)
axes[0].set_xlabel("Configuration", fontsize=16, fontweight="bold").set_visible(False)
axes[0].set_yticklabels(
    ["{:.2f}".format(y) for y in axes[0].get_yticks()], fontsize=10, fontweight="bold"
)

# axes[0].set_title('mAP', fontsize=14, fontweight='bold')
axes[0].set_ylabel("AP", fontsize=16, fontweight="bold")
axes[0].set_xticklabels(
    mAP_combined["Model"], rotation=0, color="black", fontweight="bold"
)
axes[0].set_ylim(bottom=0.2)
axes[0].set_xlabel("Configuration", fontsize=16, fontweight="bold").set_visible(False)
axes[0].set_yticklabels(
    ["{:.2f}".format(x) for x in axes[0].get_yticks()], fontsize=10, fontweight="bold"
)
#mAP_combined['Process time'] = 30.0 /mAP_values['Process time'] 
mAP_combined[["Processing time ratio"]].plot(kind="bar", ax=axes[1], color="#ff7f0e")
# axes[1].set_title('Processing time ratio', fontsize=14, fontweight='bold')
for i, bar in enumerate(axes[1].patches):
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
    )
axes[1].set_ylabel("FPS", fontsize=16, fontweight="bold")
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
plt.savefig("/home/ubuntu/Tesis/Results/Tesis/SelectionOfModel/AP_PerModel.png")
plt.tight_layout()
plt.show()
