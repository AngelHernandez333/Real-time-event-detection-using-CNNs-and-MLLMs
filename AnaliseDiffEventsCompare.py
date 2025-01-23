import pandas as pd

df1 = pd.read_csv("Results/Analize_F1.csv")
df2 = pd.read_csv("Results/Analize_Recall.csv")
df3 = pd.read_csv("Results/Analize_Precision.csv")

shape = df1.shape
# Ensure the dataframes have the same shape and columns
assert shape == df2.shape
assert shape == df3.shape

assert all(df1.columns == df2.columns) and all(
    df3.columns == df2.columns
), "Columns do not match"

import matplotlib.pyplot as plt

# Set the index to the first column for better plotting
df1.set_index(df1.columns[0], inplace=True)
df2.set_index(df2.columns[0], inplace=True)
df3.set_index(df3.columns[0], inplace=True)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

colors = ["#2E86C1", "#F39C12", "#28B463"]
# colors=['#E74C3C', '#F1C40F', '#1ABC9C']
# colors = ['#3498DB', '#9B59B6', '#1ABC9C']

# Plot the data
df1.mean(axis=1).plot(
    kind="bar", ax=ax, position=0, width=0.25, label="F1 Score", color=colors[0]
)
df2.mean(axis=1).plot(
    kind="bar", ax=ax, position=1, width=0.25, label="Recall", color=colors[1]
)
df3.mean(axis=1).plot(
    kind="bar", ax=ax, position=2, width=0.25, label="Precision", color=colors[2]
)

# Set the labels and title
ax.set_xlabel("Categories", fontsize=12, fontweight="bold")
ax.set_xticklabels(df1.index, rotation=0, color="black", fontweight="bold", fontsize=7)
ax.set_ylabel("Scores", fontsize=12, fontweight="bold")
ax.set_title(
    "Comparison of F1 Score, Recall, and Precision as metric",
    fontsize=12,
    fontweight="bold",
)
ax.grid()

# Set the limits of the x axis
ax.set_xlim(-1.0, len(df1.index) + 0.5)

# Add a legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

print("\nF1 ", df1.mean(), "\nRecall ", df2.mean(), "\nPrecision ", df3.mean())
