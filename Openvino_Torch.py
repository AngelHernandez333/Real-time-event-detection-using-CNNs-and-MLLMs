

import openvino.torch
import torch
model = torch.compile('/home/ubuntu/yolov10/int8/yolov10x_openvino_model/', backend='openvino')
