import torch
import os
import sys
import shutil
import torch.backends.cudnn as cudnn
import argparse
import skimage.metrics
import numpy as np
from torch.utils.data import DataLoader
from utils import write_html, write_loss, get_config, write2images, get_all_data_loaders
from torch.utils.tensorboard import SummaryWriter
from models.LUM_model import DecomNet
from trainer import UNIT_Trainer

# parse options
parser = argparse.ArgumentParser(description='DenoiseNet args setting')
parser.add_argument('--denoise_config', type=str, default='configs/unit_NDM.yaml', help='Path to the config file.')
parser.add_argument('--light_config', type=str, default='configs/unit_LUM.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='./denoise', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='UNIT', help="UNIT")
parser.add_argument('--light_checkpoint', type=str, default='./checkpoints/LUM_LOL.pth',
                    help="checkpoint of light")
opts = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    cudnn.benchmark = True
    # load train setting
    denoise_config = get_config(opts.denoise_config)
    max_iter = denoise_config['max_iter']
    display_size = denoise_config['display_size']
    denoise_config['vgg_model_path'] = opts.output_path

    # pre-trained model set
    light_config = get_config(opts.light_config)
    light = DecomNet(light_config)
    state_dict = torch.load(opts.light_checkpoint, map_location='cpu')
    light.load_state_dict(state_dict)
    light.cuda()
    light.eval()

    # model set and data loader
    trainer = UNIT_Trainer(denoise_config)
    if torch.cuda.is_available():
        trainer.cuda(denoise_config['gpuID'])
        torch.nn.DataParallel(trainer)
    train_loader_x, train_loader_y, test_loader_x, test_loader_y = get_all_data_loaders(denoise_config)

    # set logger and output folder
    writer = SummaryWriter(os.path.join(opts.output_path + "/logs"))
    output_directory = os.path.join(opts.output_path + "/outputs")

    # set checkpoint folder
    checkpoint_directory = os.path.join(output_directory, 'checkpoints_denoise')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)

    # set image folder
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)

    # copy config file to output folder
    shutil.copy(opts.denoise_config, os.path.join(output_directory, 'config_yaml'))

    # start training
    psnr = 0
    ssim = 0
    count = 0
    print('start training')
    iterations = trainer.resume(checkpoint_directory, hyperparameters=denoise_config) if opts.resume else 0
    while True:
        for it, (images_x, images_y, val_x, val_y) in enumerate(zip(train_loader_x, train_loader_y, test_loader_x, test_loader_y)):
            dataX, dataY = images_x.cuda().detach(), images_y.cuda().detach()
            valX, valY = val_x.cuda().detach(), val_y.cuda().detach()
            dataX, _ = light(dataX)
            # main training code
            for _ in range(3):
                trainer.content_update(dataX, dataY, denoise_config)
            trainer.dis_update(dataX, dataY, denoise_config)
            trainer.gen_update(dataX, dataY, denoise_config)
            trainer.update_learning_rate()

            # dump training stats in log file
            if (iterations + 1) % denoise_config['log_iter'] == 0:
                write_loss(iterations, trainer, writer)
            if (iterations + 1) % denoise_config['image_save_iter'] == 0:
                trainer.eval()
                print("[*] Evaluating for phase train / epoch %d..." % (iterations + 1))
                with torch.no_grad():
                    train_image_outputs = trainer.sample(dataX, dataY)
                    val_image_outputs = trainer.sample(valX, valY)
                write2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                write2images(val_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_html(output_directory + "/index.html", iterations + 1, denoise_config['image_save_iter'],
                           'images')
                print("===> Iteration[{}]: psnr: {}, ssim:{}".format(iterations + 1, psnr / count, ssim / count))
            if (iterations + 1) % denoise_config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)
            iterations += 1
            if iterations >= max_iter:
                writer.close()
                sys.exit('Finish training')


if __name__ == '__main__':
    main()
