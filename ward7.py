import pandas as pd

rute='/home/ubuntu/Tesis/Results/Tesis/Best_CLIP/'
file='TestingCLIP16_NWPUIITB_4.csv'
df1= pd.read_csv(f"{rute}TestingCLIP_RULES16_MLLMNewPrompts.csv")
df2= pd.read_csv(f"{rute}{file}")
df= pd.concat([df1, df2], ignore_index=True)
print(df)
df.to_csv(f"{rute}TestCLIPAll4.csv", index=False)