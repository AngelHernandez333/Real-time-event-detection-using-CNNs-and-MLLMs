from CLIPS import CLIP_Model
import os
from PIL import Image

model = CLIP_Model()
model.set_model("openai/clip-vit-base-patch32")
model.set_processor("openai/clip-vit-base-patch32")

descriptions = [
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
    "a person tripping",
    "a person stealing other person's pocket",
]

# Add a prefix to each description
prefix = "a video of "
descriptions = [prefix + description for description in descriptions]
rute = f"/home/ubuntu/Tesis/Temp/"
files = os.listdir(rute)
images = []

import matplotlib.pyplot as plt

for i in range(6):
    image_path = f"/home/ubuntu/Tesis/Temp/{files[i]}"

    image = Image.open(image_path)
    images.append(image)

    # Display the image
    plt.figure()
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Image {i}")
model.set_descriptions(descriptions)
outputs = model.outputs(images)
print(outputs)
