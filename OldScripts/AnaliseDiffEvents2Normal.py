import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def softmax(z):
    """
    Compute the softmax of vector z.

    Parameters:
    z (numpy.ndarray): Input vector.

    Returns:
    numpy.ndarray: Softmax of the input vector.
    """
    exp_z = np.exp(z - np.max(z))  # Subtract max(z) for numerical stability
    return exp_z / np.sum(exp_z)


df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsLLavaAV_SameVideosDifVal3.csv")
# df['True Event']='everything is normal'
df.drop(["Mode", "Duration", "Process time"], axis=1, inplace=True)
df_else = df[df["Check event"] != "everything is normal"]
df_normal = df[df["Check event"] == "everything is normal"]
df_normal.rename(
    columns={
        "True Positive": "False Positive",
        "False Positive": "True Positive",
        "False Negative": "True Negative",
        "True Negative": "False Negative",
    },
    inplace=True,
)
print("The others\n", df_else, "\nNormal\n", df_normal)
# df_save = df[df['Check event'] != 'everything is normal']
# df_save.to_csv('Results/resultsLLavaAV_NormalVideos2.csv', index=False)

df = pd.concat([df_else, df_normal], ignore_index=True)
print(df)
"""df.rename(columns={
    'True Positive': 'False Positive',
    'False Positive': 'True Positive',
    'False Negative': 'True Negative',
    'True Negative': 'False Negative'
}, inplace=True)"""
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df.drop(
    ["True Positive", "True Negative", "False Positive", "False Negative"],
    axis=1,
    inplace=True,
)
df.fillna(0, inplace=True)

videos = df["Name"].unique()
for i in range(len(videos)):
    video = videos[i]
    df1 = df[df["Name"] == video]
    print(df1)

"""
print(df['False Positive'].sum())
df.rename(columns={
    'True Positive': 'True Negative',
    'False Positive': 'False Negative',
    'False Negative': 'False Positive',
    'True Negative': 'True Positive'
}, inplace=True)
print(df)
df['Recall'] = df['True Positive'] / (df['True Positive'] + df['False Negative'])
df= df[['Name','Recall', 'True Event','Check event']]
df.fillna(0, inplace=True)
print(df)
df_results=pd.DataFrame(columns=['Name', 'True Event', 'Predicted event'])"""
