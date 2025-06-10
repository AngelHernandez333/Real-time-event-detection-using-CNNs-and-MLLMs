import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv("/home/ubuntu/Tesis/Results/results.csv")

# Count the occurrences of each type of event

# Get unique categories
categories = df["Event"].unique()

# Separate rows by category
category_dfs = {category: df[df["Event"] == category] for category in categories}

# Example: Access DataFrame for category
for i in range(len(categories)):
    df1 = category_dfs[categories[i]]
    #
    names = df1["Name"].unique()
    grouped = df1.groupby("Name")
    max = grouped[["Validations Number"]].max()
    # df1[['Validations Number']] = scaler.fit_transform(df1[['Validations Number']])

    for name in names:
        df1.loc[df1["Name"] == name, "Validations Number"] = (
            df1.loc[df1["Name"] == name, "Validations Number"]
            / max.loc[name, "Validations Number"]
        )
    grouped = df1.groupby("Mode")
    mean_values = grouped[["Precision", "Recall", "Validations Number"]].mean()
    mean_values["F1"] = (
        2
        * (mean_values["Precision"] * mean_values["Recall"])
        / (mean_values["Precision"] + mean_values["Recall"])
    )
    mean_values = mean_values[["F1", "Validations Number"]]
    # Plot the results
    mode_names = {0: "Decision Maker Complex", 1: "Only MLLM", 2: "MLLM and Detector"}
    mean_values.rename(index=mode_names, inplace=True)
    mean_values.plot(kind="bar")
    plt.title(f"{categories[i]}")
    plt.xlabel("Mode")
    plt.ylabel("Mean Values")
    plt.xticks(rotation=45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid()
plt.show()
