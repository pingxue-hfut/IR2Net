'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch
import math
from modules import ir_1w1a


__all__ = ['vgg_small_1w1a']


class CIRec(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(CIRec, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)   # m=init_channels, ratio=s, oup=n
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.Hardtanh(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.Hardtanh(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.nonlinear = nn.Hardtanh()

        self.conv1 = ir_1w1a.IRConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = ir_1w1a.IRConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = ir_1w1a.IRConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = ir_1w1a.IRConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = ir_1w1a.IRConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.fusion = CIRec(640, 512, ratio=32)

        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w1a.IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)   # 128
        x = self.bn0(x)
        x = self.nonlinear(x)
        fx = self.avgpool(x)

        x = self.conv1(x)   # 128
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)

        x = self.conv2(x)  # 256
        x = self.bn2(x)
        x = self.nonlinear(x)

        x = self.conv3(x)  # 256
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)

        x = self.conv4(x)  # 512
        x = self.bn4(x)
        x = self.nonlinear(x)

        x = self.conv5(x)  # 512
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        feature = x

        x = torch.cat([fx, x], dim=1)
        x = self.fusion(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, feature


def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model
