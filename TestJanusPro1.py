
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import cv2
import time
import numpy as np
# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-1B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

question='Tell if it is a video or not, and if it is a video, what is the video about?'
use_image=["1.png", "1.png"
        ,"1.png", "1.png",
        "1.png",]
images_number=len(use_image)*'<image_placeholder>'
assert len(use_image) > 4, "Please provide at least 4 images"
conversation = [
    {
        "role": "<|User|>",
        "content": f"{images_number}  This is a video, \n{question}",
        "images": [],
    },
    {"role": "<|Assistant|>", "content": ""},
]
# Function to convert OpenCV (BGR) to PIL (RGB)
def cv2_to_pil(cv_image):
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image_rgb)
    return pil_image

# Load images with PIL and convert to RGB
images=[cv2.imread(image_path) for image_path in use_image]
start_time=time.time()
pil_images = [cv2_to_pil(image) for image in images]
end_time=time.time()-start_time
print('Time taken to load images:', end_time)


# load images and prepare for inputs
#pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

# # run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# # run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
#rint(f"{prepare_inputs['sft_format'][0]}", answer)
print('Here\n\n\n', answer)
