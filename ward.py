import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the CSV file into a DataFrame
df = pd.read_csv('/home/ubuntu/Tesis/Results/results.csv')

# Count the occurrences of each type of event

# Get unique categories
categories = df['Event'].unique()

# Separate rows by category
category_dfs = {category: df[df['Event'] == category] for category in categories}

# Example: Access DataFrame for category
for i in range(len(categories)):
    df1 = category_dfs[categories[i]]
    #
    names =df1['Name'].unique()
    grouped = df1.groupby('Name')
    max=grouped[['Validations Number']].max()
    for name in names:
        df1.loc[df1['Name'] == name, 'Validations Number'] = (
            df1.loc[df1['Name'] == name, 'Validations Number'] / max.loc[name, 'Validations Number']
        )
    print(df1)
    grouped = df1.groupby('Mode')
    #----------------------------------------------------------------------
    def calculate_ap(precision, recall):
        sorted_indices = np.argsort(-recall)
        precision = precision[sorted_indices]
        recall = recall[sorted_indices]
        print('Precision:', precision, 'Recall:', recall)
        ap = 0.0
        for i in range(len(recall)):
            if i == 0:
                ap += precision[i] * recall[i]
            else:
                ap += precision[i] * (recall[i] - recall[i - 1])
        print('AP:', ap)

        return ap
    # Calculate AP for each mode
    ap_values = {}
    for mode, group in grouped:
        precision = group['Precision'].values
        recall = group['Recall'].values
        print('Mode:', mode, '\n')
        ap = calculate_ap(precision, recall)
        ap_values[mode] = ap
    mean_values = grouped[['Precision', 'Recall', 'Validations Number']].mean()
plt.figure()
y=[0.83333333, 0.87878788, 0.33076923] 
x= [0.27027027, 0.4084507 , 0.64179104]
y=np.array(y)
x=np.array(x)
plt.plot(x, y, marker='o')



plt.show()

