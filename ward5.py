import numpy as np

'''import numpy as np

frame1 = dict(np.load("../Database/NWPU_IITB/GT/gt_Avenue.npz"))  # Convert NpzFile to a dictionary
for key in list(frame1.keys()):
    new_key = key.split('.')[0]  # Split the key by the comma and take the first part
    frame1[new_key] = frame1[key]  # Add the new key-value pair
    del frame1[key]  # Remove the old key
for key in frame1.keys():
    print(key)
for key in frame1.keys():
    print(frame1[key])
np.savez("../Database/NWPU_IITB/GT/gt_newavenue.npz", **frame1)'''

'''import numpy as np 

frame1 = np.load("../Database/NWPU_IITB/GT/gt_IITB.npz")
frames2 = np.load("../Database/NWPU_IITB/GT/NWPU_Campus_gt.npz")

frames3 = np.load("../Database/NWPU_IITB/GT/gt_newavenue.npz")

# Combine the two dictionaries of arrays
combined_frames = {**frame1, **frames2, **frames3}

# Save the combined arrays into a new .npz file
np.savez("../Database/NWPU_IITB/GT/gt.npz", **combined_frames)
frame1 = dict(np.load("../Database/NWPU_IITB/GT/gt.npz"))  # Convert NpzFile to a dictionary
for key in frame1.keys():
print(key)'''
def calculate_ap(precision, recall):
    # Sort by recall (ascending)
    sorted_indices = np.argsort(recall)
    precision = np.array(precision)[sorted_indices]
    recall = np.array(recall)[sorted_indices]

    # Pad with (0,0) and (1,0)
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))

    # Compute AP as the area under the raw curve (no interpolation)
    ap = 0.0
    for i in range(1, len(recall)):
        delta_recall = recall[i] - recall[i-1]
        ap += delta_recall * precision[i]

    return ap

import pandas as pd
import numpy as np
frame1 = dict(np.load("/home/ubuntu/Tesis/Results/TestingJanusScore.npz"))
df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingJanusScore.csv")
print(frame1.keys())

categories = df["True Event"].unique()
category_dfs = {category: df[df["True Event"] == category] for category in categories}

mAP_process = []
for i in range(len(categories)):
    df1 = category_dfs[categories[i]]
    #
    df1["Process time"] = df1["Process time"] / df1["Duration"]
    print(df1, categories[i])
    modes= df1["Mode"].unique()
    for mode in modes:
        gt=np.array([])
        scores=np.array([])
        df2 = df1[df1["Mode"] == mode]
        print(df2, mode)
        for index, row in df2.iterrows():
            video = row["Name"]
            video=video.split(".")[0]
            video_score=frame1[f'{video}_{mode}']
            video_gt=frame1[f'{video}_{mode}_gt']
            scores=np.concatenate((scores, video_score))
            gt=np.concatenate((gt, video_gt))
        print(scores.shape, gt.shape)

    # Get unique scores as thresholds (sorted descending)
    thresholds = np.unique(scores)[::-1]

    precision = []
    recall = []

    # Apply each threshold
    for thresh in thresholds:
        # Predict 1 if score >= threshold, 0 otherwise
        predictions = (scores >= thresh).astype(int)
        
        # Compute TP, FP, FN
        tp = np.sum((predictions == 1) & (gt == 1))
        fp = np.sum((predictions == 1) & (gt == 0))
        fn = np.sum((predictions == 0) & (gt == 1))
        
        # Calculate precision and recall
        prec = tp / (tp + fp + 1e-10)  # Avoid division by zero
        rec = tp / (tp + fn + 1e-10)
        
        precision.append(prec)
        recall.append(rec)

    # Compute AP
    ap = calculate_ap(precision, recall)
    print(f"Average Precision (AP): {ap:.4f}")

    # Optional: Print precision-recall pairs for inspection
    for t, p, r in zip(thresholds, precision, recall):
        print(f"Threshold: {t:.2f}, Precision: {p:.4f}, Recall: {r:.4f}")