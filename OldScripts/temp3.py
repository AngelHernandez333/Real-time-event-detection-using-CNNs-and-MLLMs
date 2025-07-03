import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("/home/ubuntu/Tesis/Results/TestingJanusCLIP.csv")
categories = df["True Event"].unique()
category_dfs = {category: df[df["True Event"] == category] for category in categories}

for i in range(len(categories)):
    df1 = category_dfs[categories[i]]
    check_event_counts = {}
    for _, row in df1.iterrows():
        check_event = row["Check event"]
        if pd.isna(check_event):
            continue
        else:
            check_event = check_event.split(",")
            for string in check_event:
                print(string.split(":"), int(string.split(":")[1]))
                if string.split(":")[0] in check_event_counts:
                    check_event_counts[string.split(":")[0]] += int(
                        string.split(":")[1]
                    )
                else:
                    check_event_counts[string.split(":")[0]] = int(string.split(":")[1])
    print(check_event_counts)

    # Plot the dictionary as a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(check_event_counts.keys(), check_event_counts.values(), color="skyblue")
    plt.title(f"Event Counts for True Event: {categories[i]}")
    plt.xlabel("Check Event")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()
