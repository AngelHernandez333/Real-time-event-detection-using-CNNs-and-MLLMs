'''import pandas as pd
df1 = pd.read_csv("/home/ubuntu/Tesis/Results/TestingJanusAllOnlyTrue.csv")
df1=df1[df1['Mode']==0]
print(df1['Process time'].sum()/(60*60))

print(df1[df1['True Positive']==0])'''
import numpy as np 

frame1 = np.load("../Database/NWPU_IITB/GT/gt_IITB.npz")
frames2 = np.load("../Database/NWPU_IITB/GT/NWPU_Campus_gt.npz")

frames3 = np.load("../Database/NWPU_IITB/GT/gt_newavenue.npz")

# Combine the two dictionaries of arrays
combined_frames = {**frame1, **frames2, **frames3}

# Save the combined arrays into a new .npz file
np.savez("../Database/NWPU_IITB/GT/gt.npz", **combined_frames)
frame1 = dict(np.load("../Database/NWPU_IITB/GT/gt.npz"))  # Convert NpzFile to a dictionary
for key in frame1.keys():
    print(key)