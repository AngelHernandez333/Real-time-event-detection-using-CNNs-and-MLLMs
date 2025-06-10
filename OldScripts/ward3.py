import pandas as pd

df= pd.read_csv("/home/ubuntu/Tesis/Results/Tesis/PromptSelection/TestingJanusConfirm.csv")
df1= pd.read_csv("/home/ubuntu/Tesis/Results/Tesis/PromptSelection/TestingJanusTell_IsThere.csv")
df=df[df["Mode"]==1]
df1= df1[df1["Mode"]==1]
df1 = df1[df1["Name"].isin(df["Name"])]
df1 = df1[df1["True Event"].isin(df["True Event"])]
print(df, df1)
df1.to_csv("/home/ubuntu/Tesis/Results/Tesis/PromptSelection/TestingJanusTell_IsThere.csv")