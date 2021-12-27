import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import ToTensor


class SingleDatasetFromFolder(data.Dataset):
    def __init__(self, x_dir, gt_dir):
        super(SingleDatasetFromFolder, self).__init__()
        self.image_filenames_x = [os.path.join(x_dir, x) for x in os.listdir(x_dir) if is_image_file(x)]
        self.image_filenames_gt = [os.path.join(gt_dir, x) for x in os.listdir(gt_dir) if is_image_file(x)]

    def __getitem__(self, index):
        x_image = Image.open(self.image_filenames_x[index])
        gt_image = Image.open(self.image_filenames_gt[index])
        return ToTensor()(x_image), ToTensor()(gt_image)

    def __len__(self):
        return len(self.image_filenames_x)


class ValDatasetFromFolder(data.Dataset):
    def __init__(self, x_dir, y_dir, gt_dir):
        super(ValDatasetFromFolder, self).__init__()
        self.image_filenames_x = [os.path.join(x_dir, x) for x in os.listdir(x_dir) if is_image_file(x)]
        self.image_filenames_y = [os.path.join(y_dir, x) for x in os.listdir(y_dir) if is_image_file(x)]
        self.image_filenames_gt = [os.path.join(gt_dir, x) for x in os.listdir(gt_dir) if is_image_file(x)]

    def __getitem__(self, index):
        x_image = Image.open(self.image_filenames_x[index])
        y_image = Image.open(self.image_filenames_y[index])
        gt_image = Image.open(self.image_filenames_gt[index])
        return ToTensor()(x_image), ToTensor()(y_image), ToTensor()(gt_image)

    def __len__(self):
        return len(self.image_filenames_y)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
