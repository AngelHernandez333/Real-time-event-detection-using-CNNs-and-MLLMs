import pandas as pd

df = pd.read_csv("/home/ubuntu/Tesis/Results/Jumping1video.csv")
print(df)
df['Porcentage']=df['Mean']/(df['Height'])
print(df['Porcentage'].max(), df['Porcentage'].min())