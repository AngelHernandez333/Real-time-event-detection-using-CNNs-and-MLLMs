from abc import ABC, abstractmethod
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import numpy as np


class MLLMs(ABC):
    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    def set_processor(self, processor):
        pass

    @abstractmethod
    def event_validation(self):
        pass


class LLaVA_OneVision(MLLMs):
    def __init__(self):
        self.__model = None
        self.__processor = None

    def set_model(self, model):
        self.__model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(dtype=torch.float16, device="cuda")

    def set_processor(self, processor):
        self.__processor = AutoProcessor.from_pretrained(processor)

    def event_validation(self, frames, event, text="Watch the video,", verbose=False):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {
                        "type": "text",
                        "text": f"{text} there is {event}? Just yes or no",
                    },
                ],
            },
        ]
        video = np.stack(frames)
        prompt = self.__processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.__processor(
            videos=list(video), text=prompt, return_tensors="pt"
        ).to("cuda:0", torch.float16)
        out = self.__model.generate(**inputs, max_new_tokens=60)
        text_outputs = self.__processor.batch_decode(
            out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if verbose:
            print(text_outputs[0].split("\n")[-1])
        return text_outputs[0].split("\n")[-1]


if __name__ == "__main__":
    print(__name__)
