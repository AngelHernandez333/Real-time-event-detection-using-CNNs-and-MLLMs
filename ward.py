'''import pandas as pd

df = pd.read_csv("/home/ubuntu/Tesis/Results/Falling12video.csv")
print(df)


df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsOOP2.csv")
print(df)
df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df.fillna(0, inplace=True)
print(df[["Name", "Mode", "Precision", "Recall"]])'''
"""precision_sum = df["Precision"].mean()
recall_sum = df["Recall"].mean()

df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsOOP1.csv")
print(df)
df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df.fillna(0, inplace=True)
print(df[["Precision", "Recall"]])
precision_mean = df["Precision"].mean()
recall_mean = df["Recall"].mean()

print(precision_sum, recall_sum, precision_mean, recall_mean)
print("Diference Precision:", precision_sum - precision_mean)
print("Diference Recall:", recall_sum - recall_mean)"""

import pandas as pd

df = pd.read_csv("/home/ubuntu/Tesis/VideosEventAllDuration.csv")
print(df)
print(df['Duration'].sum()/(60*60)) 