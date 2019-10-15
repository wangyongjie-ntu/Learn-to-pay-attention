#Filename:	vgg19.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Mon 07 Oct 2019 08:39:38 PM

import torch
import torch.nn as nn


model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, num_classes, init_weights = False):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.conv_block1 = self._conv_block(3, 64, 2)
        self.conv_block2 = self._conv_block(64, 128, 3)
        self.conv_block3 = self._conv_block(128, 256, 3)
        self.conv_block4 = self._conv_block(256, 512, 3)
        self.conv_block5 = self._conv_block(512, 512, 3)
        self.conv_block6 = self._conv_block(512, 512, 2, is_pool = True)


    def forward(self, input_x):
        input_x = self.conv_block1(input_x)
        input_x = self.conv_block2(input_x)
        input_x = self.conv_block3(input_x)
        input_x = self.conv_block4(input_x)
        input_x = self.conv_block5(input_x)
        input_x = self.conv_block6(input_x)
        return input_x

    def _initialize_weights(self):
        NotImplemented
        return

    def _conv_block(self, input_depth, num_filters, num_layers, is_pool = False):
        layers = []
        layers.append(nn.Conv2d(input_depth, num_filters, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        layers.append(nn.BatchNorm2d(num_filters, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace = True))

        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
            layers.append(nn.BatchNorm2d(num_filters, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace = True))

        if is_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        else:
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)))

        return nn.Sequential(*layers)


if __name__ == "__main__":
    test = VGG(100)
    input_x = torch.randn((1, 3, 224, 224))
    print(test(input_x).shape)

