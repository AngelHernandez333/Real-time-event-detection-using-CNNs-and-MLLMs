import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


"""df = pd.read_csv('/home/ubuntu/Tesis/Results/resultsLLavaAV_SameVideosDifVal3.csv')
# Sum the 'Duration' column and convert to hours
total_duration_seconds = df['Process time'].sum()
total_duration_hours = total_duration_seconds / 3600
print(f"Total Duration in hours: {total_duration_hours/6}")"""
"""df.rename(columns={
    'True Positive': 'True Negative',
    'False Positive': 'False Negative',
    'False Negative': 'False Positive',
    'True Negative': 'True Positive'
}, inplace=True)"""
"""df.rename(columns={
    'True Positive': 'False Positive',
    'False Positive': 'True Positive',
    'False Negative': 'True Negative',
    'True Negative': 'False Negative'
}, inplace=True)"""


"""df.drop(['Mode','Validations Number','Duration', 'Process time'], axis=1, inplace=True)
df.rename(columns={
    'True Positive': 'True Negative',
    'False Positive': 'False Negative',
    'False Negative': 'False Positive',
    'True Negative': 'True Positive'
}, inplace=True)
df['Recall'] = df['True Positive'] / (df['True Positive'] + df['False Negative'])
df.drop([ 'True Positive','True Negative',
    'False Positive','False Negative'], axis=1, inplace=True)"""
for k in range(1, 5):  # Pasar por todos los modos
    print(f"Mode {k}")

df_normal = pd.read_csv("/home/ubuntu/Tesis/Results/resultsLLavaAV_NormalVideos.csv")
df_normal["True Event"] = "everything is normal"
df_normal.to_csv("Results/resultsMode0Samevideos_Normals.csv", index=False)
