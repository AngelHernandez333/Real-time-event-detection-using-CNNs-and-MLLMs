from abc import ABC, abstractmethod
import time
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel, XCLIPModel
import torch
from PIL import Image
import cv2
import numpy as np


class CLIP(ABC):
    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    def set_processor(self, processor):
        pass

    @abstractmethod
    def outputs(self):
        pass

    @abstractmethod
    def get_descriptions(self, descriptions):
        pass

    def set_descriptions(self, descriptions):
        pass


class CLIP_Model(CLIP):
    def __init__(self):
        self.__model = None
        self.__processor = None
        self.__descriptions = None

    def set_model(self, model):
        device = "cuda"
        torch_dtype = torch.float16

        self.__model = CLIPModel.from_pretrained(
            model,
            attn_implementation="sdpa",
            device_map=device,
            torch_dtype=torch_dtype,
        )

    def set_processor(self, processor):
        self.__processor = CLIPProcessor.from_pretrained(processor)

    def set_descriptions(self, descriptions):
        self.__descriptions = descriptions

    def get_descriptions(self):
        return self.__descriptions

    def outputs(self, images):
        device = "cuda"
        torch_dtype = torch.float16
        start = time.time()
        # pil_images = [
        #    cv2_to_pil(frame) for frame in images]
        inputs = self.__processor(
            text=self.__descriptions, images=images, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():

            with torch.autocast(device):

                outputs = self.__model(**inputs)

        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score

        probs = logits_per_image.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilities
        max_probs, max_indices = probs.max(dim=1)
        # Print the max probability and corresponding description for each image
        for i in range(len(max_probs)):
            print(
                f"Image {i}: Max probability: {max_probs[i].item()}, Description: {self.__descriptions[max_indices[i].item()]}"
            )
        # Calculate the average probability for each description
        avg_probs = probs.mean(dim=0)
        # Find the description with the highest average probability
        max_avg_prob, max_avg_index = avg_probs.max(dim=0)
        print(
            f"\n\n\nHighest average probability: {max_avg_prob.item()}, Description: {self.__descriptions[max_avg_index.item()]}"
        )
        print(f"\nTime {time.time()-start} seconds")
        return self.__descriptions[max_avg_index.item()], max_avg_prob.item()


class XCLIP_Model(CLIP):
    def __init__(self):
        self.__model = None
        self.__processor = None
        self.__descriptions = None

    def set_model(self, model):
        device = "cuda"
        torch_dtype = torch.float16
        self.__model = AutoModel.from_pretrained(
            model, torch_dtype=torch_dtype, device_map=device
        )

    def set_processor(self, processor):
        device = "cuda"
        torch_dtype = torch.float16
        self.__processor = AutoProcessor.from_pretrained(processor)

    def set_descriptions(self, descriptions):
        self.__descriptions = descriptions

    def get_descriptions(self):
        return self.__descriptions

    def outputs(self, images, padding):
        device = "cuda"
        torch_dtype = torch.float16
        start = time.time()
        if len(padding) == 1:
            frames = padding + padding + images
        else:
            frames = padding + images
        print(f"Cantidad de frames {len(frames)}")
        result = np.stack(frames)
        print(f"Descripciones {self.__descriptions}")
        inputs = self.__processor(
            text=self.__descriptions,
            videos=list(result),
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():

            with torch.autocast(device):

                outputs = self.__model(**inputs)

        logits_per_video = (
            outputs.logits_per_video
        )  # this is the image-text similarity score

        probs = logits_per_video.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilities
        max_probs, max_indices = probs.max(dim=1)
        # Print the max probability and corresponding description for each image
        for i in range(len(max_probs)):
            print(
                f"Image {i}: Max probability: {max_probs[i].item()}, Description: {self.__descriptions[max_indices[i].item()]}"
            )
        # Calculate the average probability for each description
        avg_probs = probs.mean(dim=0)
        # Find the description with the highest average probability
        max_avg_prob, max_avg_index = avg_probs.max(dim=0)
        print(
            f"\n\n\nHighest average probability: {max_avg_prob.item()}, Description: {self.__descriptions[max_avg_index.item()]}"
        )
        print(f"\nTime {time.time()-start} seconds")
        return self.__descriptions[max_avg_index.item()], max_avg_prob.item()

    def outputs_without_softmax(self, images, padding):
        device = "cuda"
        torch_dtype = torch.float16
        start = time.time()
        if len(padding) == 1:
            frames = padding + padding + images
        else:
            frames = padding + images
        print(f"Cantidad de frames {len(frames)}")
        result = np.stack(frames)
        print(f"Descripciones {self.__descriptions}")
        inputs = self.__processor(
            text=self.__descriptions,
            videos=list(result),
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():

            with torch.autocast(device):

                outputs = self.__model(**inputs)

        logits_per_video = (
            outputs.logits_per_video
        )  # this is the image-text similarity score

        probs = logits_per_video.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilities
        max_probs, max_indices = probs.max(dim=1)
        # Print the max probability and corresponding description for each image
        for i in range(len(max_probs)):
            print(
                f"Image {i}: Max probability: {max_probs[i].item()}, Description: {self.__descriptions[max_indices[i].item()]}"
            )
        # Calculate the average probability for each description
        avg_probs = probs.mean(dim=0)
        # Find the description with the highest average probability
        max_avg_prob, max_avg_index = avg_probs.max(dim=0)
        print(
            f"\n\n\nHighest average probability: {max_avg_prob.item()}, Description: {self.__descriptions[max_avg_index.item()]}"
        )
        print(f"\nTime {time.time()-start} seconds")
        print(logits_per_video)
        return (
            self.__descriptions[max_avg_index.item()],
            max_avg_prob.item(),
            logits_per_video.to(dtype=torch.float32, device="cpu").numpy()[0],
        )
