from abc import ABC, abstractmethod
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import cv2


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
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image_rgb)
        return pil_image


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
        #attn_implementation="flash_attention_2",
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
                        "text": f"{text} is there {event}? Just yes or no"
                        ,
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

    def event_validation(self, frames, event, text="Watch the video,", verbose=False):
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
                "content": f"{images_number} This is a video \n{text} tell me if in the video is there {event}? Just yes or no",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        '''conversation = [
            {
                "role": "<|User|>",
                "content": f"{images_number} This is a video \n{text}, there is a person pickpocketing another person, tell me how would you describe that action so you can understand it better",
                "images": [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]'''
        # Load images with PIL and convert to RGB

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


if __name__ == "__main__":
    print(__name__)
