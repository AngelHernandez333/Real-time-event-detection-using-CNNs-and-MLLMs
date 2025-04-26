import pandas as pd

df= pd.read_csv("/home/ubuntu/Tesis/Results/Tesis/PerformanceNewPrompt/TestingNWPUIITB.csv")
print(df)
print(df['Process time'].sum()/(60*60))