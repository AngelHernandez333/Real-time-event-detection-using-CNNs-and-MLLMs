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
df = pd.read_csv("/home/ubuntu/Tesis/Results/results29Nov.csv")

# Count the occurrences of each type of event

# Get unique categories
categories = df["Event"].unique()

# Separate rows by category
category_dfs = {category: df[df["Event"] == category] for category in categories}


mAP_process = []
for i in range(len(categories)):
    df1 = category_dfs[categories[i]]
    #
    names = df1["Name"].unique()
    grouped = df1.groupby("Name")
    max = grouped[["Validations Number"]].max()
    for name in names:
        df1.loc[df1["Name"] == name, "Validations Number"] = (
            df1.loc[df1["Name"] == name, "Validations Number"]
            / max.loc[name, "Validations Number"]
        )
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
    mean_values = grouped[["Precision", "Recall", "Validations Number"]].mean()
    print("Average Precision (AP) for each mode:")
    for mode, ap in ap_values.items():
        print(f"Mode {mode}: {ap:.4f}")
    mean_values["AP"] = [ap_values[mode] for mode in mean_values.index]
    mean_values = mean_values[["AP", "Validations Number"]]
    mAP_process.append(mean_values)
    # Plot the results
    mode_names = {0: "Decision Maker Complex", 1: "Only MLLM", 2: "MLLM and Detector"}
    mean_values.rename(index=mode_names, inplace=True)
    mean_values.plot(kind="bar")
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
mAP_values[["mAP"]].plot(kind="bar")
plt.title(f"mAP")
plt.xlabel("Mode")
plt.ylabel("mAP")
plt.xticks(rotation=45)
plt.legend(loc="best")
plt.tight_layout()
plt.grid()
plt.show()
