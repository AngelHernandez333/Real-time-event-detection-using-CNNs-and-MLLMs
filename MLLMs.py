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
import random


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

    def event_selection(
        self, frames, descriptions, text="Watch the video and", verbose=False
    ):
        number_of_frames = 4
        assert (
            len(frames) >= number_of_frames
        ), f"Please provide at least {number_of_frames} frames."
        # 5
        images_number = (
            len(frames[-(1 + number_of_frames) : -1]) * "<image_placeholder>"
        )
        random.shuffle(descriptions)
        descriptions_text = "\n".join(
            f"{i+1}. {desc}" for i, desc in enumerate(descriptions)
        )

        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} select the one of the following descriptions:
{descriptions_text}\nJust the number of the selected description.""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} select the description that best describes the video from the following descriptions:
{descriptions_text}\nJust the number of the selected description.""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # Shuffle the descriptions list randomly
        '''random.shuffle(descriptions)
        descriptions_text = "\n".join(f"- {desc}" for i, desc in enumerate(descriptions))
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} select the description that best describes the video from the following descriptions:
{descriptions_text}\n
Return the selected description without any explanation or extra text in the following format:
The selected description is: [selected_description].""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]'''
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} select the description that best describes the video from the following descriptions:
{descriptions_text}\n
The selecting MUST not be based on the order of the descriptions, only in the content of the description and the video.

Return the selected description without any explanation or extra text in the following format:
The selected description is: [selected_description].""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} select the description that best describes the video from the following descriptions:
{descriptions_text}\n
The selecting MUST not be based on the order of the descriptions, only in the content of the description and the video.

Return THE EXACT AND UNIQUE selected description without any explanation or extra text in the following format:
The selected description is: [selected_description].""",
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
        # return answer[-4:-1]  # Extract the score from the answer
        return answer

    def event_score(self, frames, event, text="Watch the video and", verbose=False):
        number_of_frames = 4
        assert (
            len(frames) >= number_of_frames
        ), f"Please provide at least {number_of_frames} frames."
        # 5
        images_number = (
            len(frames[-(1 + number_of_frames) : -1]) * "<image_placeholder>"
        )

        # -----------------------------------------
        descriptions_text = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(event))

        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} evaluate how well each of the following descriptions matches the video by providing a score from 0 to 100 for each.
{descriptions_text}. Ensure there are a score for each of the {len(event)} descriptions, the exact number.
Think that each score must be independent of the others, only matters the description and the video.

Respond in the following format:
Scores:
1. [score1]
2. [score2]
...""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        random.shuffle(event)
        descriptions_text = "\n".join(f"- {desc}" for i, desc in enumerate(event))
        descriptions_score = "\n".join(
            f"- {desc} Score: [score]" for i, desc in enumerate(event)
        )
        descriptions_score = "\n".join(
            f"- {desc} Score: [selected score]" for i, desc in enumerate(event)
        )

        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} evaluate how well each of the following descriptions matches the video by providing a confidence score from 0 to 100 for each.\n
{descriptions_text}\n 
Ensure there are a score for each of the {len(event)} descriptions, the exact number and follow the order.
Think that each score must be independent of the others, only matters the description and the video.

Respond in the following format:
Scores:
- [description][score1]
- [description][score2]
...

Until you finish the score for each description, MUST be the exact number of scores.""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} evaluate how well each of the following descriptions matches the video by providing a confidence score from 0 to 100 for each.\n
{descriptions_text}\n 
Ensure there are a score for each of the {len(event)} descriptions, the exact number.
Think that each score must be independent of the others, only matters the description and the video.

Respond in the following format:
Scores:
- [description][score1]
- [description][score2]
...
Return the score for each description, MUST be the exact number of descriptions.""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} evaluate how well each of the following descriptions matches the video by providing a confidence score from 0 to 100 for each.\n
{descriptions_text}\n 
Ensure there are a score for each of the {len(event)} descriptions, the exact number.
Think that the score of each descriptions MUST be independent of the others, determine it only analizing the description and the video.

Respond in the following format:
Scores:
- [description][score1]
- [description][score2]
...
Return the score for each description, MUST be the exact number of descriptions.""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} evaluate how well each of the following descriptions matches the video by providing a confidence score from 0 to 100 for each.\n
{descriptions_text}\n 
Ensure there are a score for each of the {len(event)} descriptions, the exact number.
Think that the score of each descriptions MUST be independent of the others, determine it only analizing the description and the video.

Respond in the following format:
Scores:
{descriptions_score}
...
Return the score for each description, MUST be the exact number of descriptions.""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video.
{text} evaluate how well each of the following descriptions matches the video by providing a confidence score from 0 to 100 for each.\n
{descriptions_text}\n 
Ensure there are a score for each of the {len(event)} descriptions, the exact number.
Think that the score of each descriptions MUST be independent of the others and of the order,  determine it only analizing the description and the video.

Respond in the following format:
Scores:
{descriptions_score}
...
Return the score for each description, MUST be the exact number of descriptions.""",
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
            # print(answer)
        # return answer[-4:-1]  # Extract the score from the answer
        return answer

    def event_validation_score(
        self, frames, event, text="Watch the video.", verbose=False
    ):
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
                "content": f"{images_number} This is a video \n{text} Does the video contain {event}? Just yes or no.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text} Does the video contain {event}? Just yes or no with a confident score from 0 to 100.",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # Scores muy binarios 0 or 100
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?. Just yes or no.
                Support your answer with a confident score from 0 to 100.
                Return the answer in the following format:
                Answer: [answer]
                Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # Seems ok, need test it, the No answer just give 0
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?. Just yes or no.
                Support your answer with a confident score from 0 to 100, where from 0 to 50 is for No, and from 50 to 100 is for Yes, intermediate values are allowed.
                Return the answer in the following format:
                Answer: [answer]
                Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?. Just yes or no.
                Support your answer with a confident score from 0 to 100 (intermediate values are allowed), where from 0 to 50 is for No, and from 50 to 100 is for Yes.
Return the answer in the following format:
Answer: [answer]
Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?. Just yes or no.
Support your answer with a confident score from 0 to 100 (intermediate values are allowed, with increments of 5), where from 0 to 50 is for No, and from 50 to 100 is for Yes.
Return the answer in the following format:
Answer: [answer]
Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?.  
Just yes or no.
Support your answer with a confident score from 0 to 100, where from 0 to 50 is for No, and from 50 to 100 is for Yes. Intermediate values are allowed.
Return the answer in the following format:
Answer: [answer]
Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]  # Repeats the answer a several number of times, seems ok the answers but some answers have brackets
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a surveillance video.\n{text}
Does the video contain {event}? Just yes or no.

Look closely at all parts of the video, including the edges and background.

Support your answer with a confident score from 0 to 100, where from 0 to 50 is for No, and from 50 to 100 is for Yes. Intermediate values are allowed.

Return the answer in the following format:
Answer: [answer]
Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]  # Repeats the answer a several number of times, seems ok the answers but some answers have brackets
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?.  
Just yes or no.
Look closely at all parts of the video, including the edges and background.
Support your answer with a confident score from 0 to 100, where from 0 to 50 is for No, and from 50 to 100 is for Yes. Intermediate values are allowed.
Return the answer in the following format:
Answer: [answer]
Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # The best until now
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?.  
Just yes or no.
Support your answer with a confident score from 0 to 100, where from 0 to 50 is for No, and from 50 to 100 is for Yes. Intermediate values are allowed.
Return the answer in the following format:
Answer: [answer]
Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # **Best until now**

        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?.  
Just yes or no.
Look closely at all parts of the video, including the edges and background.
Support your answer with a confident score from 0 to 100, where from 0 to 50 is for No, and from 50 to 100 is for Yes. Intermediate values are allowed.
Return the answer in the following format:
Answer: [answer]
Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # Test
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?.  
Just yes or no.
Look closely at all parts of the video, including the edges and background.
Support your answer with a confident score from 0 to 100, where 0 to 50 indicates No (lower values for higher confidence in No), and 50 to 100 indicates Yes (higher values for higher confidence in Yes). Intermediate values are allowed.
Return the answer in the following format:
Answer: [answer]
Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?.  
Just yes or no.
Look closely at all parts of the video, including the edges and background.
Support your answer with a confident score from 0 to 100, where 0 to 50 indicates No (lower values for higher confidence in No), and 50 to 100 indicates Yes (higher values for higher confidence in Yes). Intermediate values are allowed.
Return a single answer in the following format (without brackets):
Answer: [answer]
Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        # OK in format, but the scores are still just some of them, not all the range,
        # Also, sometimes dont see the event present
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a surveillance video.\n{text}
Does the video contain {event}? Just yes or no.
- Provide only one answer.
- The answer must be either 'Yes' or 'No' without brackets or other variations.
- The score must be a number between 0 and 100, reflecting the confidence in your answer. Use 0 to 50 for No (lower values indicate higher confidence in No) and 50 to 100 for Yes (higher values indicate higher confidence in Yes).
- Choose a precise score based on the evidence in the video. Avoid defaulting to extreme values (0, 50, 100) unless the evidence is absolutely conclusive. Intermediate values are strongly preferred to reflect partial confidence.
- Do not include any additional text, explanations, or repeated answers.
- Examine all parts of the video, including edges and background, before answering.
Respond with exactly one answer in the following format:
Answer: [Yes or No]
Score: [score]
""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?.  
Just yes or no.
Look closely at all parts of the video, including the edges and background.
Support your answer with a confident score from 0 to 100, where from 0 to 50 is for No, and from 50 to 100 is for Yes. Intermediate values are allowed.
Choose a precise score based on the evidence in the video. Avoid defaulting to extreme values (0, 50, 100) unless the evidence is absolutely conclusive. Intermediate values are strongly preferred to reflect partial confidence.
Do not include any additional text, explanations, or repeated answers.
Return the answer in the following format:
Answer: [answer]
Score: [score]""",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        conversation = [
            {
                "role": "<|User|>",
                "content": f"""{images_number} This is a video \n{text} Does the video contain {event}?.  
Just yes or no.
Look closely at all parts of the video, including the edges and background.
Support your answer with a confident score on a scale from 0 to 100, where from 0 to 50 is for No, and from 50 to 100 is for Yes. Intermediate values are allowed.
Choose a precise score based on the video. Avoid defaulting to extreme values (0, 50, 100) unless the evidence is absolutely conclusive. Intermediate values are strongly preferred to reflect partial confidence.
Do NOT include any additional text, explanations, or repeated answers.
Return the answer in the following format:
Answer: [answer]
Score: [score]""",
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
            max_new_tokens=30,
            do_sample=False,
            use_cache=True,
        )

        answer = self.__tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        # rint(f"{prepare_inputs['sft_format'][0]}", answer)
        if verbose:
            print(f"{prepare_inputs['sft_format'][0]}", answer)

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
