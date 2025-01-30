'''import pandas as pd

df = pd.read_csv("/home/ubuntu/Tesis/Results/Falling12video.csv")
print(df)


df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsOOP2.csv")
print(df)
df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df.fillna(0, inplace=True)
print(df[["Name", "Mode", "Precision", "Recall"]])'''
"""precision_sum = df["Precision"].mean()
recall_sum = df["Recall"].mean()

df = pd.read_csv("/home/ubuntu/Tesis/Results/resultsOOP1.csv")
print(df)
df["Precision"] = df["True Positive"] / (df["True Positive"] + df["False Positive"])
df["Recall"] = df["True Positive"] / (df["True Positive"] + df["False Negative"])
df.fillna(0, inplace=True)
print(df[["Precision", "Recall"]])
precision_mean = df["Precision"].mean()
recall_mean = df["Recall"].mean()

print(precision_sum, recall_sum, precision_mean, recall_mean)
print("Diference Precision:", precision_sum - precision_mean)
print("Diference Recall:", recall_sum - recall_mean)"""
import cv2
import numpy as np
from PIL import Image
import timeit

# Load an image using OpenCV
cv_image = cv2.imread("1.png")

# Function to convert OpenCV (BGR) to PIL (RGB)
def cv2_to_pil():
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image_rgb)
    return pil_image

# Measure execution time
time_taken = timeit.timeit(cv2_to_pil, number=100) / 100  # Average over 100 runs
print(f"Average conversion time: {time_taken * 1000:.3f} ms")
