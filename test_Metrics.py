from PIL import Image
import numpy as np
import os
import skimage.metrics
from torchvision import transforms
from utils import load_images, load_images_no_norm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test(image_path, gt_path):
    filelist = os.listdir(testfolder)
    psnr_total = 0
    ssim_total = 0
    count = 0
    for i, file in enumerate(filelist):
        print(file)
        testmat = Image.open(os.path.join(testfolder + '/' + file)).convert('RGB')
        testmat = np.array(testmat)
        gtmat = Image.open(os.path.join(gtfolder + '/' + file)).convert('RGB')
        gtmat = np.array(gtmat)
        # testmat = load_images_no_norm(os.path.join(testfolder + '/' + file))
        # gtmat = load_images_no_norm(os.path.join(gtfolder + '/' + file))
        # w, h, _ = testmat.shape
        # gtmat = gtmat[0:w, 0:h, :]
        psnr = skimage.metrics.peak_signal_noise_ratio(testmat, gtmat)
        ssim = skimage.metrics.structural_similarity(testmat, gtmat, multichannel=True)
        psnr_total += psnr
        ssim_total += ssim
        print(ssim)
        print(psnr)
        count += 1
    print(count)
    print('mean psnr: {}, ssim:{}'.format(psnr_total / count, ssim_total / count))

if __name__ == "__main__":
    testfolder = './NDM_LOL/'
    gtfolder = '/home/dancer/LOL/test_gt'
    test(testfolder, gtfolder)
