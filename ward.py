import pandas as pd
df1 = pd.read_csv("/home/ubuntu/Tesis/Results/Meeting/mAPJanuOldPrompt.csv")
df2= pd.read_csv("/home/ubuntu/Tesis/Results/Meeting/mAPJanusNewPrompt.csv")

print('Information of classes', df1, '\nExtra information of the classes', df2)
