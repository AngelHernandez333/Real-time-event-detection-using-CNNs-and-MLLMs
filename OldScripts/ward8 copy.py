import pandas as pd

# New prompt
rute = "/home/ubuntu/Tesis/Results/Tesis/PerformanceNewPrompt/"
file = "TestingNWPUIITB.csv"
df1 = pd.read_csv(f"{rute}{file}")
file = "TestingJanusPrompts.csv"
df2 = pd.read_csv(f"{rute}{file}")
df1["Process time"] = df1["Process time"] / df1["Duration"]
df2["Process time"] = df2["Process time"] / df2["Duration"]
df = pd.concat([df1, df2], ignore_index=True)
df = df2
storing_file = "ALLwithRespectedFPS.png"

# df = df[(df['Mode'] == 0) | (df['Mode'] == 2)]
#  Get unique categories
categories = df["True Event"].unique()
# Separate rows by category
df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df.fillna(0, inplace=True)
category_dfs = {category: df[df["True Event"] == category] for category in categories}

mAP_process = []
for i in range(len(categories)):
    df1 = category_dfs[categories[i]]
    #
    print(categories[i])
    grouped = df1.groupby("Mode")
    mean_values = grouped[["Precision", "Recall", "Process time"]].mean()
    mode_names = {
        0: "Detector con reglas, MLLM e información",
        1: "MMLM Solo",
        2: "Detector, MLLM e informacion",
        3: "Detector con reglas y MLLM",
        4: "Detector con reglas ",
    }
    mean_values.rename(index=mode_names, inplace=True)
    mAP_process.append(mean_values)
    print(mean_values)
mAP_values = pd.concat(mAP_process).groupby(level=0).mean()
mAP_values["Process time"] = 30.0 / mAP_values["Process time"]
print("Final\n\n ", mAP_values["Process time"])
