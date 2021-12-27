from __future__ import print_function
from utils import get_config
from trainer import UNIT_Trainer
import argparse
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
from models.LUM_model import DecomNet

parser = argparse.ArgumentParser()
parser.add_argument('--denoise_config', type=str, default='./configs/unit_NDM.yaml', help="denoise net configuration")
parser.add_argument('--light_config', type=str, default='configs/unit_LUM.yaml', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, default='./test_images', help="input image path")
parser.add_argument('--output_folder', type=str, default='./NDM_results', help="output image path")
parser.add_argument('--denoise_checkpoint', type=str, default='./checkpoints/NDM_LOL.pt', help="checkpoint of denoise")
parser.add_argument('--light_checkpoint', type=str, default='./checkpoints/LUM_LOL.pth', help="checkpoint of light")
opts = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
denoise_config = get_config(opts.denoise_config)

# Setup model and data loaderoots.trainer == 'UNIT':
DN_trainer = UNIT_Trainer(denoise_config)
state_dict = torch.load(opts.denoise_checkpoint, map_location='cpu')
DN_trainer.gen_x.load_state_dict(state_dict['x'])
DN_trainer.gen_y.load_state_dict(state_dict['y'])
DN_trainer.cuda()
DN_trainer.eval()
encode = DN_trainer.gen_x.encode_cont  # encode function
decode = DN_trainer.gen_y.decode_cont  # decode function

# pre-trained model set
light_config = get_config(opts.light_config)
light = DecomNet(light_config)
state_dict = torch.load(opts.light_checkpoint, map_location='cpu')
light.load_state_dict(state_dict)
light.cuda()
light.eval()


if not os.path.exists(opts.input_folder):
    raise Exception('input path is not exists!')
imglist = os.listdir(opts.input_folder)
transform = transforms.Compose([transforms.ToTensor()])

for i, file in enumerate(imglist):
    print(file)
    with torch.no_grad():
        filepath = opts.input_folder + '/' + file
        image = transform(Image.open(filepath).convert('RGB')).unsqueeze(0).cuda()
        # Start testing
        h, w = image.size(2), image.size(3)
        pad_h = h % 4
        pad_w = w % 4
        image = image[:, :, 0:h - pad_h, 0:w - pad_w]
        r_low, i_low = light(image)
        content = encode(r_low)
        outputs = decode(content)
        if not os.path.exists(opts.output_folder):
            os.makedirs(opts.output_folder)
        outputs_back = outputs.clone()
        name = os.path.splitext(file)[0]
        path = os.path.join(opts.output_folder, name + '.png')
        vutils.save_image(outputs.data, path)
