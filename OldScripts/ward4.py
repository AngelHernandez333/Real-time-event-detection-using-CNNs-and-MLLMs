import pandas as pd


df = pd.read_csv(
    "/home/ubuntu/Tesis/Results/Tesis/SelectionOfModel/TestingJanusIsThere.csv"
)
df2 = pd.read_csv(
    "/home/ubuntu/Tesis/Results/Tesis/SelectionOfModel/TestingIsThereLlava.csv"
)
df3 = pd.read_csv(
    "/home/ubuntu/Tesis/Results/Tesis/SelectionOfModel/TestingIsThereQwen.csv"
)

events = df["True Event"].unique()
print(events)
df1 = df[df["Mode"] == 1]
df2 = df2[df2["Mode"] == 1]
df3 = df3[df3["Mode"] == 1]
df2 = df2[df2["True Event"].isin(events)]
df3 = df3[df3["True Event"].isin(events)]
print(df2)
print(df3)
print(df1)

df2.to_csv(
    "/home/ubuntu/Tesis/Results/Tesis/SelectionOfModel/TestingIsThereLlava.csv",
    index=False,
)
df3.to_csv(
    "/home/ubuntu/Tesis/Results/Tesis/SelectionOfModel/TestingIsThereQwen.csv",
    index=False,
)

count = df1.groupby("True Event").count()
print(count)
