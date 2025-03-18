
'''import pandas as pd
df=pd.read_csv('/home/ubuntu/Tesis/Results/TestingJanusAll.csv')
print(df)
print(df['Duration'].sum()/(60*60))
print(df['Process time'].sum()/(60*60))
#df.drop(columns='Unnamed: 0', inplace=True)
#print(df)
#df.to_csv('/home/ubuntu/Tesis/Results/TestingJanusAll.csv', index=False)
df=df[df['Mode']==0]
df=df.groupby('Name').count()
print(df[df['Mode']!=13])'''
'''import numpy as np 
import os
rute = f"/home/ubuntu/Database/CHAD DATABASE/CHAD_Meta/anomaly_labels/"
files = os.listdir(rute)
number=0
for i in range(len(files)):
    temp=np.load(rute+files[i])
    if temp.sum()==0.0:
        pass
    else:
        number+=temp.shape[0]
print(number)'''
import pandas as pd

df = pd.read_csv('/home/ubuntu/Tesis/PredictedMode0.csv')

# Group by 'Name' and sum the 'Corrects' column
df = df.groupby('Name', as_index=False)['Correct'].sum()
print(df)
print('Correct precision:', df['Correct'].sum()/df.shape[0])