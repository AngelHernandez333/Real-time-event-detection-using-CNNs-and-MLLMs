import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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


def calculate_map(name, rute):
    df = pd.read_csv(f"{rute}/{name}")
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
    return mAP_values


if __name__ == "__main__":
    rute = "/home/ubuntu/Tesis/Results/temp"
    files = os.listdir(rute)
    results = []
    for i in range(len(files)):
        results.append(calculate_map(files[i], rute))
    print("\n\n\n")
    final_results = pd.DataFrame()
    for i in range(len(results)):
        result = results[i]
        result["File"] = files[i].split(".")[0].split("TestingJanus")[-1]
        final_results = pd.concat([final_results, result])
    print(final_results)
    # Plot the AP values for each file
    file_names = {
        "Is": "is 'event' in the video?",
        "Tell_IsThere": "tell me if in the video is there 'event'?",
        "Tell_Is": "tell me if is 'event' in the video? ",
        "Does": "does the video contain 'event'?",
        "IsThere": "is there 'event'?",
        "Confirm": "confirm if the video contain 'event'",
    }
    final_results["File"] = final_results["File"].map(file_names)

    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(final_results["File"].unique()))
    final_results = final_results.sort_values(by="AP", ascending=True)
    for i, mode in enumerate(final_results.index.unique()):
        subset = final_results.loc[mode]
        plt.bar(index + i * bar_width, subset["AP"], bar_width, label=mode)
        for j, value in enumerate(subset["AP"]):
            plt.text(
                index[j] + i * bar_width,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                color="black",
                fontweight="bold",
            )

    plt.xlabel("Prompt").set_visible(False)
    plt.ylabel("AP", fontsize=13, fontweight="bold")
    plt.title(
        "AP de cada prompt", fontsize=16, fontweight="bold"
    )
    plt.xticks(
        index + bar_width / 2,
        final_results["File"].unique(),
        rotation=45,
        color="black",
        fontweight="bold",fontsize=11,
    )
    plt.yticks(
        fontsize=10, fontweight="bold"
    )
    plt.legend().set_visible(False)
    plt.grid(True)
    plt.tight_layout(pad=4.0)
    plt.ylim(bottom=0.6, top=0.8)
    print(final_results)
    plt.savefig('/home/ubuntu/Tesis/Results/Tesis/PromptSelection/AP_PerPrompt.png')
    plt.show()
