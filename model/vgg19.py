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


model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, im_size, num_classes, init_weights = "kaimingNormal", attention = False):
        super(VGG, self).__init__()
        self.im_size = im_size
        self.num_classes = num_classes
        self.init_weights = init_weights
        self.attention = attention
        self.conv_block1 = self._conv_block(3, 64, 2)
        self.conv_block2 = self._conv_block(64, 128, 3)
        self.conv_block3 = self._conv_block(128, 256, 3)
        self.conv_block4 = self._conv_block(256, 512, 3)
        self.conv_block5 = self._conv_block(512, 512, 3)
        self.conv_block6 = self._conv_block(512, 512, 2, is_pool = True)
        self.adaptive_conv = nn.Conv2d(512, 512, kernel_size = int(im_size / 32), padding = 0, bias = True)
        # project to the query dimension
        self.projector = nn.Conv2d(256, 512, kernel_size = 1, padding = 0, bias = False)
        self.op = nn.Conv2d(512, 1, kernel_size = 1, padding = 0, bias = False) 

        if self.attention:
            self.classifier = nn.Linear(512 * 3, self.num_classes, bias = True)
        else:
            self.classifier = nn.Linear(512, self.num_classes, bias = True)
        
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

        if self.attention:
            l1 = self.projector(l1)
            c1, g1 = self.linearAttentionBlock(l1, g)
            c2, g2 = self.linearAttentionBlock(l2, g)
            c3, g3 = self.linearAttentionBlock(l3, g)
            g = torch.cat((g1, g2, g3), dim = 1)
            x = self.classifier(g)
        else:
            c1, c2, c3 = None, None, None
            x = self.classifier(torch.squeeze(g))

        return [x, c1, c2, c3]

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

    def linearAttentionBlock(self, l, g, normlize_method = "sigmoid"):
        N, C, W, H = l.size()
        c = self.op(l + g)
        if normlize_method == "softmax":
            a = F.softmax(c.view(N, -1, 1), dim =  2).view(N, 1, W, H)
        elif normlize_method == "sigmoid":
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        
        if normlize_method == "softmax":
            g = g.view(N, C, -1).sum(dim = 2)
        elif normlize_method == "sigmoid":
            g = F.adaptive_avg_pool2d(g, (1, 1)).view(N, C)

        return c.view(N, 1, W, H), g

    # Reference: Attention-Gated Networks https://arxiv.org/abs/1804.05338 & https://arxiv.org/abs/1808.08114 
    def gridAttentionBlock(self, l, g, normlize_method):
        NotImplemented

if __name__ == "__main__":
    test = VGG(224, 100, attention = True)
    input_x = torch.randn((1, 3, 224, 224))
    output_x = test(input_x)
    print(output_x[0].shape, output_x[1].shape, output_x[2].shape, output_x[3].shape)

