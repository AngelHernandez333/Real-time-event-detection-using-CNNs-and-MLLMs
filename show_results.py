import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("/home/ubuntu/Tesis/Results/mAP_values6events64videos.csv")#64 videos
df2=pd.read_csv("/home/ubuntu/Tesis/Results/mAP_values6events.csv")#83 videos
df3=pd.read_csv("/home/ubuntu/Tesis/Results/mAP_values8events.csv")#99 videos
df.index = df["Mode"]
df2.index = df2["Mode"]
df3.index = df3["Mode"]
df=df.drop(columns=['Mode'])
df2=df2.drop(columns=['Mode'])
df3=df3.drop(columns=['Mode'])

# Plotting the AP column comparison between df and df2
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the AP values
bar_width = 0.35
index = np.arange(len(df.index))

bar1 = ax.bar(index, df['AP'], bar_width, label='64 videos')
bar3 = ax.bar(index + bar_width, df3['AP'], bar_width, label='83 videos')

# Adding labels, title, and legend
ax.set_xlabel('Mode')
ax.set_ylabel('AP')
ax.set_title('Comparison of AP values between 64 and 99 videos')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(df.index, rotation=45)
ax.legend()

plt.tight_layout()
plt.show()