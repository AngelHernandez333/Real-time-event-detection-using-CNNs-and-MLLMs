"""import pandas as pd

df = pd.read_csv("/home/ubuntu/Tesis/Results/Falling12video.csv")
print(df)


df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsOOP2.csv")
print(df)
df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df.fillna(0, inplace=True)
print(df[["Name", "Mode", "Precision", "Recall"]])"""

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

"""df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsMode1_5Samevideos.csv")
df2 = pd.read_csv("/home/ubuntu/Tesis/Results/resultsMode0Samevideos.csv")
df=pd.concat([df,df2])
#print(df)
#print(df['Duration'].sum()/(60*60)) 
ward=df[df['True Event']==df['Check event']] 
events=ward['True Event'].unique()
print(events)
running=ward[ward['True Event']==events[3]] 
lying=ward[ward['True Event']==events[4]] 
save=pd.concat([running,lying])
save.to_csv("/home/ubuntu/Tesis/Results/RunningLyingOld.csv",index=False)"""
df = pd.read_csv("/home/ubuntu/Tesis/Results/mAP_valuesNew.csv")
df2 = pd.read_csv("/home/ubuntu/Tesis/Results/mAP_valuesOld.csv")
print("Nuevas reglas", df, "\nOld rules", df2)

import numpy as np

df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsTest.csv")
print(df[df["Validations Number"] == 0])
df2 = pd.read_csv("/home/ubuntu/Tesis/Results/RunningLyingOld.csv")
print(df[df["Validations Number"] == 0])
ward = [188.59, 198.68, 203.53, 201.51, 203.05, 181.95, 177.74]
ward_array = np.array(ward)
ward_array = np.array(ward)
is_increasing = np.all(np.diff(ward_array) > 0)
is_decreasing = np.all(np.diff(ward_array) < 0)
print("Increasing:", is_increasing)
print("Decreasing:", is_decreasing)

ward = np.array(ward)
print(np.mean(ward), np.std(ward))
