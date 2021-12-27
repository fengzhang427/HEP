from __future__ import print_function
from utils import get_config
from trainer import UNIT_Trainer
import matplotlib.pyplot as plt
import argparse
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
from models.LUM_model import DecomNet

parser = argparse.ArgumentParser(description='Light args setting')
parser.add_argument('--LUM_config', type=str, default='configs/unit_LUM.yaml', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, default='./test_images',
                    help="input image path")
parser.add_argument('--output_folder', type=str, default='./LUM_results', help="output image path")
parser.add_argument('--LUM_checkpoint', type=str, default='./checkpoints/LUM_LOL.pth',
                    help="checkpoint of light")
opts = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

light_config = get_config(opts.LUM_config)
light = DecomNet(light_config)
state_dict = torch.load(opts.LUM_checkpoint, map_location='cpu')
light.load_state_dict(state_dict)
light.cuda()
light.eval()

if not os.path.exists(opts.input_folder):
    raise Exception('input path is not exists!')
imglist = os.listdir(opts.input_folder)
transform = transforms.Compose([transforms.ToTensor()])

for i, file in enumerate(imglist):
    print(file)
    filepath = opts.input_folder + '/' + file
    image = transform(Image.open(
        filepath).convert('RGB')).unsqueeze(0).cuda()
    # Start testing
    h, w = image.size(2), image.size(3)
    pad_h = h % 4
    pad_w = w % 4
    image = image[:, :, 0:h - pad_h, 0:w - pad_w]
    r_low, i_low = light(image)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)
    outputs_back = r_low.clone()
    name = os.path.splitext(file)[0]
    path = os.path.join(opts.output_folder, name + '.png')
    vutils.save_image(r_low.data, path)
