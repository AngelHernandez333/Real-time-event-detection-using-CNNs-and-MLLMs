from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

import os

descriptions= [
"a person riding a bicycle",
"a certain number of persons fighting",
"a group of persons playing",
"a person running",
"a person lying in the floor",
"a person chasing other person",
"a person jumping",
"a person falling",
"a person guiding other person",
"a person stealing other person",
"a person throwing trash in the floor",
'a person tripping',
"a person stealing other person's pocket",
]
rute = f"/home/ubuntu/Tesis/Temp/"
files = os.listdir(rute)
images=[]
import matplotlib.pyplot as plt

for i in range(6):
    image_path = f"/home/ubuntu/Tesis/Temp/{files[i]}"

    image = Image.open(image_path)
    images.append(image)

    # Display the image
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Image {i}")

import time 

start= time.time()
inputs = processor(text=descriptions, images=images, return_tensors="pt", padding=True)

outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

print(probs)# Get the max probability and its index for each image
max_probs, max_indices = probs.max(dim=1)

# Calculate the average probability for each description
avg_probs = probs.mean(dim=0)

# Find the description with the highest average probability
max_avg_prob, max_avg_index = avg_probs.max(dim=0)

# Print the max probability and corresponding description for each image
for i in range(len(max_probs)):
    print(f"Image {i}: Max probability: {max_probs[i].item()}, Description: {descriptions[max_indices[i].item()]}")

# Print the description with the highest average probability
print(f"\n\n\nHighest average probability: {max_avg_prob.item()}, Description: {descriptions[max_avg_index.item()]}")
print(f'\nTime {time.time()-start} seconds')
#probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
#max_prob, max_index = probs.max(dim=1)  # get the max probability and its index
#print(f"Max probability: {max_prob.item()}, Index: {descriptions[max_index.item()]}")
#plt.show() 