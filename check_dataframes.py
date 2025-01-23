import numpy as np
import pandas as pd

df = pd.read_csv("/home/ubuntu/Tesis/Results/ChaseValidations2.csv")
df["Intervals mean"] = np.abs(df["Intervals mean"])
df["Percent"] = df["Intervals mean"] / df["Distance"]
df["Status"] = 0.0
print(df)
categories = df["Video"].unique()
# Separate rows by category
category_dfs = {category: df[df["Video"] == category] for category in categories}


def check_precision(dataframe, name):
    # True positive Prediction and Reality are true
    # True negative Prediction and Reality are false
    # False negative Prediction is false and Reality is true
    # False positive Prediction is true and Reality is false
    name = name.split(".")[0]
    frames = np.load(f"../Database/CHAD DATABASE/CHAD_Meta/anomaly_labels/{name}.npy")
    for index, row in dataframe.iterrows():
        # print(dataframe.loc[index]['Frame'])
        dataframe.loc[index, "Status"] = frames[dataframe.loc[index, "Frame"] - 1]
    return dataframe


max_p = []
min_p = []
max_n = []
min_n = []
mean = []
mean_n = []
for i in range(len(categories)):
    df1 = category_dfs[categories[i]]
    df1 = check_precision(df1, categories[i])
    df1_n = df1[df1["Status"] == 0]
    df1_p = df1[df1["Status"] == 1]
    max_p.append(df1_p["Percent"].max())
    max_n.append(df1_n["Percent"].max())
    min_p.append(df1_p["Percent"].min())
    min_n.append(df1_n["Percent"].min())
    mean.append(df1_p["Percent"].mean())
    mean_n.append(df1_n["Percent"].mean())
    print(i, min_n)
max_p = np.array(max_p)
min_p = np.array(min_p)
max_n = np.array(max_n)
min_n = np.array(min_n)
mean = np.array(mean)
mean_n = np.array(mean_n)
print(max_p.mean(), min_p.mean(), "\n")

print(max_n, min_n, "\n")

print(mean.mean(), mean_n.mean(), "\n")
