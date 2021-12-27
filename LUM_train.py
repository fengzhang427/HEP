import torch
import os
import time
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import skimage.metrics
from glob import glob
from torch.utils.data import DataLoader
from utils import get_config, write_images, data_augmentation, load_images, histeq, load_vgg19
from models.LUM_model import DecomNet
from models.loss import IS_loss, Perceptual_loss
from model_dataset import SingleDatasetFromFolder

# parse options
parser = argparse.ArgumentParser(description='Light args setting')
parser.add_argument('--light_config', type=str, default='configs/unit_LUM.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='./light', help="outputs path")
parser.add_argument("--resume", action="store_true")

opts = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    # load train setting
    light_config = get_config(opts.light_config)
    max_iter = light_config['max_iter']
    display_size = light_config['display_size']
    batch_size = light_config['batch_size']
    learning_rate = light_config['lr']
    height = light_config['crop_image_height']
    width = light_config['crop_image_width']
    beta1 = light_config['beta1']
    beta2 = light_config['beta2']

    # model set and data loader
    light = DecomNet(light_config)
    if torch.cuda.is_available():
        light.cuda(light_config['gpuID'])
        torch.nn.DataParallel(light)
        cudnn.benchmark = True

    # set output folder
    output_directory = os.path.join(opts.output_path + "/outputs")

    # set checkpoint folder
    checkpoint_directory = os.path.join(output_directory, 'checkpoints_light')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)

    # set image folder
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)

    # start training
    print('start training')

    # train set
    optim = torch.optim.Adam(light.parameters(), lr=learning_rate, weight_decay=light_config['weight_decay'],
                             betas=(beta1, beta2))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[20, 40], gamma=0.1)
    criterion_I = IS_loss()
    criterion_per = Perceptual_loss()
    start_time = time.time()
    vgg = load_vgg19(20)
    if torch.cuda.is_available():
        vgg.cuda(light_config['gpuID'])
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    train_low_data = []
    train_low_data_hist = []
    train_low_data_names = glob(os.path.join(light_config['train_root'], '*.png'))
    train_low_data_names.sort()
    print('[*] Number of training data: %d' % len(train_low_data_names))
    for idx in range(len(train_low_data_names)):
        low_im = load_images(train_low_data_names[idx])
        train_low_data.append(low_im)
        train_hist_data = histeq(low_im)
        train_low_data_hist.append(train_hist_data)
    numBatch = len(train_low_data) // int(batch_size)
    image_id = 0
    psnr = 0
    mssim = 0
    ssim = 0
    count = 0
    for epoch in range(max_iter):
        for batch_id in range(numBatch):
            # generate data for a batch
            batch_input_low = np.zeros((batch_size, height, width, 3), dtype="float32")
            batch_input_low_hist = np.zeros((batch_size, height, width, 3), dtype="float32")

            for patch_id in range(batch_size):
                h, w, _ = train_low_data[image_id].shape
                x = np.random.randint(0, h - height)
                y = np.random.randint(0, w - width)
                rand_mode = np.random.randint(0, 7)
                batch_input_low[patch_id, :, :, :] = data_augmentation(
                    train_low_data[image_id][x: x + height, y: y + width, :], rand_mode)
                batch_input_low_hist[patch_id, :, :, :] = data_augmentation(
                    train_low_data_hist[image_id][x: x + height, y: y + width, :], rand_mode)
                image_id = (image_id + 1) % len(train_low_data)
                if image_id == 0:
                    tmp = list(zip(train_low_data, train_low_data))
                    np.random.shuffle(list(tmp))
                    train_low_data, _ = zip(*tmp)
            optim.zero_grad()
            low_im = torch.from_numpy(batch_input_low).cuda().permute(0, 3, 1, 2)
            low_im_hist = torch.from_numpy(batch_input_low_hist).cuda().permute(0, 3, 1, 2)
            r_low, i_low = light(low_im)
            i_low_3 = torch.cat((i_low, i_low, i_low), 1)
            recon_loss = torch.mean(torch.abs((r_low * i_low_3) - low_im))
            loss_per = criterion_per(vgg, r_low, low_im_hist)
            loss = recon_loss + 0.1 * loss_per + 0.1 * criterion_I(i_low, low_im)
            loss.backward()
            optim.step()
            print(" Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f, lr: %.6f" % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss, optim.param_groups[0]['lr']))
        scheduler.step()

        if (epoch + 1) % light_config['eval_iter'] == 0:
            val_set = SingleDatasetFromFolder(light_config['val_root'],
                                              light_config['gt_root'])
            val_loader = DataLoader(dataset=val_set, num_workers=5, batch_size=1, shuffle=False)
            light.eval()
            print("[*] Evaluating for phase train / epoch %d..." % (epoch + 1))
            with torch.no_grad():
                for val_x, val_gt in val_loader:
                    val_x = val_x.cuda()
                    val_gt = val_gt.cuda()
                    r_x, i_x = light(val_x)
                    val_gt_m = val_gt.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    val_gt_m = (val_gt_m * 255.0).round()
                    val_gt_m = val_gt_m.astype(np.uint8)
                    r_x_m = r_x.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    r_x_m = (r_x_m * 255.0).round()
                    r_x_m = r_x_m.astype(np.uint8)
                    w, h, _ = r_x_m.shape
                    val_gt_m = val_gt_m[0:w, 0:h, :]
                    psnr += skimage.metrics.peak_signal_noise_ratio(r_x_m, val_gt_m)
                    ssim += skimage.metrics.structural_similarity(r_x_m, val_gt_m, multichannel=True)
                    count += 1
                sample = val_x, r_x, i_x
                write_images(sample, display_size, '%s/sample_%s.jpg' % (image_directory, 'eval_%08d' % (epoch + 1)))
                print("===> Iteration[{}]: psnr: {}, ssim:{}".format(epoch + 1, psnr / count, ssim / count))
        if (epoch + 1) % light_config['snapshot_save_iter'] == 0:
            torch.save(light.state_dict(), checkpoint_directory + '/LUM_' + str(epoch + 1) + '.pth')

        epoch += 1


if __name__ == '__main__':
    main()
