import pandas as pd
df1 = pd.read_csv("/home/ubuntu/Tesis/Results/TestingJanusAllOnlyTrue.csv")
df1=df1[df1['Mode']==0]
print(df1['Process time'].sum()/(60*60))

print(df1[df1['True Positive']==0])