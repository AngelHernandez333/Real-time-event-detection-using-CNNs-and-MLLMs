from abc import ABC, abstractmethod
import time
from transformers import CLIPProcessor, CLIPModel


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


class CLIP_Model(CLIP):
    def __init__(self):
        self.__model = None
        self.__processor = None
        self.__descriptions = None

    def set_model(self, model):
        self.__model = CLIPModel.from_pretrained(model)

    def set_processor(self, processor):
        self.__processor = CLIPProcessor.from_pretrained(processor)

    def set_descriptions(self, descriptions):
        self.__descriptions = descriptions

    def outputs(self, images):

        start = time.time()
        inputs = self.__processor(
            text=self.__descriptions, images=images, return_tensors="pt", padding=True
        )

        outputs = self.__model(**inputs)

        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score

        probs = logits_per_image.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilities

        print(probs)  # Get the max probability and its index for each image
        # Calculate the average probability for each description
        avg_probs = probs.mean(dim=0)
        # Find the description with the highest average probability
        max_avg_prob, max_avg_index = avg_probs.max(dim=0)
        print(
            f"\n\n\nHighest average probability: {max_avg_prob.item()}, Description: {self.__descriptions[max_avg_index.item()]}"
        )
        return self.__descriptions[max_avg_index.item()]
