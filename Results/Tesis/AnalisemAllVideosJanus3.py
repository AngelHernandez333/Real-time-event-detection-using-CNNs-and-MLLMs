import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        delta_recall = recall[i] - recall[i-1]
        ap += delta_recall * precision[i]

    return ap
status='prompt_extra'
if status == 'ALL':
    rute='/home/ubuntu/Tesis/Results/Tesis/PerformanceNewPrompt/'
    file='TestingNWPUIITB.csv'
    storing_file = file.split(".")[0] + "_mAP.png"
    df = pd.read_csv(f"{rute}{file}")
    #file='TestingJanusPrompts.csv'
    file='TestingJanusPrompts.csv'
    df2 = pd.read_csv(f"{rute}{file}")
    storing_file = file.split(".")[0] + "_mAP.png"
    df["Process time"] = df["Process time"] / df["Duration"]
    df2["Process time"] = df2["Process time"] / df2["Duration"]
    df['Process time'] = 25.0 /df['Process time'] 
    df2['Process time'] = 30.0 /df2['Process time'] 
    df= pd.concat([df, df2], ignore_index=True)
    storing_file ="ALLwithRespectedFPS.png"
elif status == 'prompt_simple':
    rute='/home/ubuntu/Tesis/Results/Tesis/PerformanceOldPrompt/'
    file='TestingJanusAllOnlyTrue.csv'
    storing_file = file.split(".")[0] + "_mAP02.png"
    df = pd.read_csv(f"{rute}{file}")
    df["Process time"] = df["Process time"] / df["Duration"]
    df['Process time'] = 30.0 /df['Process time']
    df = df[(df['Mode'] == 0) | (df['Mode'] == 2)]
elif status =='prompt_extra':
    rute='/home/ubuntu/Tesis/Results/Tesis/PerformanceNewPrompt/'
    file='TestingJanusPrompts.csv'
    df2 = pd.read_csv(f"{rute}{file}")
    storing_file = file.split(".")[0] + "_mAP02.png"
    df2["Process time"] = df2["Process time"] / df2["Duration"]
    df2['Process time'] = 30.0 /df2['Process time']
    df2 = df2[(df2['Mode'] == 0) | (df2['Mode'] == 2)] 
    df=df2
elif status =='chad':
    rute='/home/ubuntu/Tesis/Results/Tesis/PerformanceNewPrompt/'
    file='TestingJanusPrompts.csv'
    df2 = pd.read_csv(f"{rute}{file}")
    storing_file = file.split(".")[0] + "_mAP.png"
    df2["Process time"] = df2["Process time"] / df2["Duration"]
    df2['Process time'] = 30.0 /df2['Process time'] 
    df=df2
elif status =='else':
    rute='/home/ubuntu/Tesis/Results/Tesis/PerformanceNewPrompt/'
    file='TestingNWPUIITB.csv'
    storing_file = file.split(".")[0] + "_mAP.png"
    df = pd.read_csv(f"{rute}{file}")
    #file='TestingJanusPrompts.csv'
    df["Process time"] = df["Process time"] / df["Duration"]
    df['Process time'] = 25.0 /df['Process time'] 
#New prompt


#df = df[(df['Mode'] == 0) | (df['Mode'] == 2)]


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
    print(df1)
    grouped = df1.groupby("Mode")
    df1['F1']= 2 * (df1['Precision'] * df1['Recall']) / (df1['Precision'] + df1['Recall'])
    df1.fillna(0, inplace=True)
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
    mean_values = grouped[["Precision", "Recall", "Process time", 'F1']].mean()
    print("Average Precision (AP) for each mode:")
    for mode, ap in ap_values.items():
        print(f"Mode {mode}: {ap:.4f}")
    mean_values["AP"] = [ap_values[mode] for mode in mean_values.index]
    mean_values = mean_values[["AP", "Process time", 'F1']]
    mAP_process.append(mean_values)
    # Plot the results
    '''mode_names = {
        0: "M5: Detector, Rules, MLLM & information",
        1: "M1: Only MLLM",
        2: "M3: Detector, MLLM & information",
        3: "M4: Detector, Rules & MLLM",
        4: "M2: Detector & Rules",
    }'''
    mode_names = {
        0: "M5: Detector con reglas, MLLM e información",
        1: "M1: MLLM Solo",
        2: "M3: Detector, MLLM e información",
        3: "M4: Detector con reglas y MLLM",
        4: "M2: Detector con reglas",
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
#mAP_values.to_csv("/home/ubuntu/Tesis/Results/Meeting/mAPJanus.csv")
mAP_values.rename(columns={"AP": "mAP"}, inplace=True)
mAP_values.rename(columns={"Process time": "Processing time ratio"}, inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

fig.suptitle("Evaluacion de desempeño: Todos los metodos", fontsize=16, fontweight="bold")

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
axes[0].set_ylabel("AP", fontsize=16, fontweight="bold")
axes[0].set_xticklabels(mAP_values.index, rotation=0, color="black", fontweight="bold", fontsize=9)
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
axes[1].set_xticklabels(mAP_values.index, rotation=0, color="black", fontweight="bold", fontsize=9)
axes[1].legend().set_visible(False)
axes[1].set_xlabel("Configuration", fontsize=16, fontweight="bold").set_visible(False)
axes[1].grid()
axes[1].set_yticklabels(
    ["{:.1f}".format(x) for x in axes[1].get_yticks()], fontsize=10, fontweight="bold"
)

fig.tight_layout(pad=3.0)
fig.set_size_inches(16, 10)

plt.tight_layout()
plt.savefig(f'{rute}{storing_file}', dpi=300, bbox_inches='tight')
plt.show()
