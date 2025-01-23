import pandas as pd

df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsLLavaAV_AllDescriptions.csv")
df.drop(["Validations Number"], axis=1, inplace=True)
print(df[df["Check event"] == "everything is normal"])
df_else = df[df["Check event"] == "everything is normal"]
print(df_else)
df_else["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
print(df_else)
