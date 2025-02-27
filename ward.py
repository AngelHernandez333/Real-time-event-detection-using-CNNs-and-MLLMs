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

'''import matplotlib.pyplot as plt
df = pd.read_csv('/home/ubuntu/Tesis/Results/resultsLLavaAV_AllDescriptions.csv')
print(df)
df=df[df['True Event']!= 'everything is normal']
df = df[df['True Event']==df['Check event']]
grouped = df.groupby('True Event').count()
print(grouped['Name'])
grouped.rename(columns={'Name':'Quantity of videos'},inplace=True)
grouped.plot(kind='bar',y='Quantity of videos', color='#000055')
plt.title('Distribution of the videos per event', fontsize=14, fontweight="bold")
plt.xlabel('Events', fontsize=12, fontweight="bold")
plt.ylabel('Quantity of videos', fontsize=12, fontweight="bold")
plt.xticks(rotation=45, fontsize=10, fontweight="bold")
plt.yticks(fontsize=10, fontweight="bold")
plt.grid()
plt.legend().set_visible(False)
plt.gcf().set_size_inches(16, 10)
plt.tight_layout()
plt.gca().set_position([0.1, 0.3, 0.8, 0.6])
plt.savefig('/home/ubuntu/Tesis/Results/DistributionVideosPerEvent.png')
plt.show()

'''
import pandas as pd

df=pd.read_csv('/home/ubuntu/Tesis/Results/TestingIsThereQwen.csv')
print(df)
print(df['Duration'].sum()/(60*60))
print(df['Process time'].sum()/(60*60))