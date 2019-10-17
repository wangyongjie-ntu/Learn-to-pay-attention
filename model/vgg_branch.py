#Filename:	vgg19.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Mon 07 Oct 2019 08:39:38 PM

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, ".")
from utils.initialize import *

class VGG(nn.Module):
    def __init__(self, im_size, num_classes, init_weights = "kaimingNormal"):
        super(VGG, self).__init__()
        self.im_size = im_size
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.conv_block1 = self._conv_block(3, 64, 2)
        self.conv_block2 = self._conv_block(64, 128, 3)
        self.conv_block3 = self._conv_block(128, 256, 3)
        self.conv_block4 = self._conv_block(256, 512, 3)
        self.conv_block5 = self._conv_block(512, 512, 3)
        self.conv_block6 = self._conv_block(512, 512, 2, is_pool = True)
        self.adaptive_conv = nn.Conv2d(512, 512, kernel_size = int(im_size / 32), padding = 0, bias = True)
        # project to the query dimension
        self.projector = nn.Conv2d(256, 512, kernel_size = 1, padding = 0, bias = False)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier1 = nn.Linear(512 * 3, self.num_classes, bias = True)
        # original fully connected layer
        self.classifier2 = nn.Linear(512, self.num_classes, bias = True)
        
        # weights initialization
        if init_weights == "kaimingUniform":
            weights_init_kaimingUniform(self)
        elif init_weights == "kaimingNormal":
            weights_init_kaimingNormal(self)
        elif init_weights == "xavierUniform":
            weights_init_kaimingNormal(self)
        elif init_weights == "xavierNormal":
            weights_init_kaimingNormal(self)
        else:
            raise NotImplementedError("Invalid type of initialization")

    def forward(self, input_x):
        input_x = self.conv_block1(input_x)
        input_x = self.conv_block2(input_x)
        l1 = self.conv_block3(input_x)
        input_x = F.max_pool2d(l1, kernel_size = 2, stride = 2, padding = 0)
        l2 = self.conv_block4(input_x)
        input_x = F.max_pool2d(l2, kernel_size = 2, stride = 2, padding = 0)
        l3 = self.conv_block5(input_x)
        input_x = F.max_pool2d(l3, kernel_size = 2, stride = 2, padding = 0)
        input_x = self.conv_block6(input_x)
        g = self.adaptive_conv(input_x)

        l1 = self.projector(l1)
        l1 = self.adaptive_avg_pool2d(l1)
        l2 = self.adaptive_avg_pool2d(l2)
        l3 = self.adaptive_avg_pool2d(l3)

        l1 = l1.view(l1.size(0), 512)
        l2 = l2.view(l2.size(0), 512)
        l3 = l3.view(l3.size(0), 512)
        g = g.view(g.size(0), 512)

        concat_l = torch.cat((l1,l2,l3), dim = 1)
        classification_1 = self.classifier1(concat_l)
        classification_2 = self.classifier2(g)

        return [classification_1, classification_2]

    def _conv_block(self, input_depth, num_filters, num_layers, is_pool = False):
        layers = []
        layers.append(nn.Conv2d(input_depth, num_filters, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        layers.append(nn.BatchNorm2d(num_filters, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace = True))
        if is_pool:
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False))

        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
            layers.append(nn.BatchNorm2d(num_filters, eps = 1e-05, momentum=0.1, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace = True))
            if is_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        return nn.Sequential(*layers)

if __name__ == "__main__":
    test = VGG(224, 100)
    input_x = torch.randn((1, 3, 224, 224))
    output_x = test(input_x)
    print(output_x[0].shape, output_x[1].shape)

