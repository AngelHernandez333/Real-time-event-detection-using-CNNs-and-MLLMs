from abc import ABC, abstractmethod
import torch
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
)
import numpy as np
from transformers import AutoModelForCausalLM
import sys
import os
from PIL import Image
import cv2

import time
from math import sqrt

# Añade la ruta de janus al sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "lib", "janus"))
from lib.janus.janus.models import MultiModalityCausalLM, VLChatProcessor


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

    @staticmethod
    def cv2_to_pil(cv_image):
        # cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        return pil_image

    @staticmethod
    def resize_frame(frame, max_pixels):
        # Calculate aspect ratio and target dimensions based on max_pixels
        h, w, _ = frame.shape
        aspect_ratio = w / h
        target_area = max_pixels
        target_h = int(sqrt(target_area / aspect_ratio))
        target_w = int(aspect_ratio * target_h)
        resized_frame = cv2.resize(
            frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC
        )
        return resized_frame


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
        # attn_implementation="flash_attention_2",
        # use_flash_attention_2=True

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
                        "text": f"{text} is there {event}? Just yes or no",
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


class JanusPro(MLLMs):
    def __init__(self):
        self.__model = None
        self.__processor = None
        self.__tokenizer = None

    def set_model(self, model):
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True
        )
        self.__model = vl_gpt.to(torch.bfloat16).cuda().eval()

    def set_processor(self, processor):
        self.__processor: VLChatProcessor = VLChatProcessor.from_pretrained(processor)
        self.__tokenizer = self.__processor.tokenizer

    def event_validation(self, frames, event, text="Watch the video.", verbose=False):
        number_of_frames = 4
        assert (
            len(frames) >= number_of_frames
        ), f"Please provide at least {number_of_frames} frames."
        # 5
        images_number = (
            len(frames[-(1 + number_of_frames) : -1]) * "<image_placeholder>"
        )
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Does the video contain {event}? Just yes or no",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]


        pil_images = [
            MLLMs.cv2_to_pil(frame) for frame in frames[-(1 + number_of_frames) : -1]
        ]

        # load images and prepare for inputs
        # pil_images = load_pil_images(conversation)
        prepare_inputs = self.__processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.__model.device)

        # # run image encoder to get the image embeddings
        inputs_embeds = self.__model.prepare_inputs_embeds(**prepare_inputs)

        # # run the model to get the response
        outputs = self.__model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.__tokenizer.eos_token_id,
            bos_token_id=self.__tokenizer.bos_token_id,
            eos_token_id=self.__tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = self.__tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        # rint(f"{prepare_inputs['sft_format'][0]}", answer)
        if verbose:
            print(f"{prepare_inputs['sft_format'][0]}", answer)

        return answer.split(".")[0]

    def event_score(self, frames, event, text="Watch the video,", verbose=False):
        number_of_frames = 4
        assert (
            len(frames) >= number_of_frames
        ), f"Please provide at least {number_of_frames} frames."
        # 5
        images_number = (
            len(frames[-(1 + number_of_frames) : -1]) * "<image_placeholder>"
        )

        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Does the video contain {event}? Answer with a score from 0.0 to 1.0",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]  # Just answer with a numerical score on a scale of 0.0 and 1.0.

        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} How would you rate the likely that the video contains {event} on a scale 0.0 to 1.0? Without any explanation or words.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} How would you rate the likely that the video contains {event} on a scale 0.0 to 1.0? Without any further explanation or previous words, just the score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Does the video contain {event}? Answer with a score on a scale from 0.0 to 1.0.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Does the video contain {event}? Answer with a score on a scale from 0.0 to 1.0 without explanations, be concise.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} How would you rate the likelihood that the video contains {event} on a scale 0.0 to 1.0?.Without any explanation or words, just score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} How would you rate the presence of {event} in the video on a scale 0.0 to 1.0?. Without any explanation or words, just score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} How would you rate the sentence '{event}' to describe the video on a scale 0.0 to 1.0?. Without any explanation or words, just score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # Between 0.5 and 0.8
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} How would you rate '{event}' to describe the video on a scale 0.0 to 1.0?. Without any explanation or words, just score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} How would you rate '{event}' to describe the content of the video on a scale 0.0 to 1.0? . Without any explanation or words, just score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Does the video contain {event}? Answer with a score on a scale from 0.0 to 1.0, without any explanation or words, just score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Does the video contain {event}? Return a score on a scale from 0.0 to 1.0, without any explanation or words, just score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # Work but gets someimes and explanations
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Does the video contain {event}? Return a score on a scale from 0.0 to 1.0 (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), without any explanation or words, just a score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Does the video contain {event}? Return a score on a scale from 0.0 to 1.0 (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), without any explanation or text, just a score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Think about the information and the video, now answer. Does the video contain {event}? Return a score on a scale from 0.0 to 1.0 (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), without any explanation or text, just a score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # Almost perfect
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Think about the information and the video, now answer with a decimal number. Does the video contain {event}? Return a score on a scale from 0.0 to 1.0 (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), without any explanation or text, just the score.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Think about the information and the video, now answer with a decimal number. Does the video contain {event}? Return a score on a scale from 0.0 to 1.0 (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0).The output must be : The score is _._.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images = [
            MLLMs.cv2_to_pil(frame) for frame in frames[-(1 + number_of_frames) : -1]
        ]

        # load images and prepare for inputs
        # pil_images = load_pil_images(conversation)
        prepare_inputs = self.__processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.__model.device)

        # # run image encoder to get the image embeddings
        inputs_embeds = self.__model.prepare_inputs_embeds(**prepare_inputs)

        # # run the model to get the response
        outputs = self.__model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.__tokenizer.eos_token_id,
            bos_token_id=self.__tokenizer.bos_token_id,
            eos_token_id=self.__tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = self.__tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        # rint(f"{prepare_inputs['sft_format'][0]}", answer)
        if verbose:
            print(f"{prepare_inputs['sft_format'][0]}", answer)
            print(answer)
        # return answer[-4:-1]  # Extract the score from the answer
        if answer[-1] == ".":
            answer = answer[-4:-1]
        return answer


class Qwen2_VL(MLLMs):
    def __init__(self):
        self.__model = None
        self.__processor = None

    def set_model(self, model):
        self.__model = Qwen2VLForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

    def set_processor(self, processor):
        # default processer
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        self.__processor = AutoProcessor.from_pretrained(
            processor, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def event_validation(self, frames, event, text="Watch the video,", verbose=False):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                    },
                    {
                        "type": "text",
                        "text": f"{text} is there {event}? Just yes or no",
                    },
                ],
            }
        ]
        video = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2).float()
        # Preparation for inference
        text = self.__processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        # image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.__processor(
            text=[text],
            videos=video,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.__model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.__processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].split(".")[0]


if __name__ == "__main__":
    print(__name__)
