from abc import ABC

from torch import nn
import torch
from models.NDM_model import Conv2dBlock
try:
    from itertools import izip as zip
except ImportError:
    pass


class DecomNet(nn.Module, ABC):
    def __init__(self, params):
        super(DecomNet, self).__init__()
        self.norm = params['norm']
        self.activ = params['activ']
        self.pad_type = params['pad_type']
        #
        self.conv0 = Conv2dBlock(4, 32, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        self.conv1 = Conv2dBlock(4, 64, 9, 1, 4, norm=self.norm, activation='none', pad_type=self.pad_type)
        self.conv2 = Conv2dBlock(64, 64, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        self.conv3 = Conv2dBlock(64, 128, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        self.conv4 = Conv2dBlock(128, 128, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        self.conv5 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.activation = nn.ReLU(inplace=True)
        self.conv6 = Conv2dBlock(128, 64, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        self.conv7 = Conv2dBlock(96, 64, 3, 1, 1, norm=self.norm, activation='none', pad_type=self.pad_type)
        self.conv8 = Conv2dBlock(64, 4, 3, 1, 1, norm=self.norm, activation='none', pad_type=self.pad_type)

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        image = torch.cat((input_max, input_im), dim=1)
        # Refelectance
        x0 = self.conv0(image)
        # print('x0:', x0.shape)
        x1 = self.conv1(image)
        # print('x1:', x1.shape)
        x2 = self.conv2(x1)
        # print('x2:', x2.shape)
        x3 = self.conv3(x2)
        # print('x3:', x3.shape)
        x4 = self.conv4(x3)
        # print('x4:', x4.shape)
        x5 = self.conv5(x4)
        x5 = self.activation(x5)
        # print('x5:', x5.shape)
        cat5 = torch.cat((x5, x2), dim=1)
        x6 = self.conv6(cat5)
        # print('x6:', x6.shape)
        cat6 = torch.cat((x6, x0), dim=1)
        x7 = self.conv7(cat6)
        # print('x7:', x7.shape)
        x8 = self.conv8(x7)
        # print('x8:', x8.shape)
        # Outputs
        R = torch.sigmoid(x8[:, 0:3, :, :])
        L = torch.sigmoid(x8[:, 3:4, :, :])
        return R, L
