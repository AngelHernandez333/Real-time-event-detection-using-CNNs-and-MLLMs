import pandas as pd

df= pd.read_csv("/home/ubuntu/Tesis/Results/Tesis/SelectionOfModel/Ward/TestingJanusIsThere.csv")
df2 = pd.read_csv("/home/ubuntu/Tesis/Results/Tesis/SelectionOfModel/Ward/TestingIsThereJanus1.csv")

df=pd.concat([df, df2], ignore_index=True)
df = df[df["Mode"] == 1]
print(df)
df.to_csv("/home/ubuntu/Tesis/Results/Tesis/SelectionOfModel/TestingJanusIsThere.csv", index=False)