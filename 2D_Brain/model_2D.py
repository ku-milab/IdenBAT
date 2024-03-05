import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
from typing import Tuple, Optional, List

from utility_2D import *



class CNNencoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU())

    def forward(self, x):
        out = self.model(x)
        return out



class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c



class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.conv1_1 = CNNencoder(in_channels, 8)
        self.conv1_2 = CNNencoder(8, 8)
        self.conv2_1 = CNNencoder(8, 16)
        self.conv2_2 = CNNencoder(16, 16)
        self.conv3_1 = CNNencoder(16, 32)
        self.conv3_2 = CNNencoder(32, 32)
        self.conv4_1 = CNNencoder(32, 64)
        self.conv4_2 = CNNencoder(64, 64)
        self.conv5_1 = CNNencoder(64, 128)
        self.conv5_2 = CNNencoder(128, 256)

    def forward(self, x):
        c1 = self.conv1_1(x) # (B, 8, 208, 176)
        c1 = self.conv1_2(c1) # (B, 8, 208, 176)
        p1 = self.pooling(c1) # (B, 8, 104, 88)

        c2 = self.conv2_1(p1) # (B, 16, 104, 88)
        c2 = self.conv2_2(c2) # (B, 16, 104, 88)
        p2 = self.pooling(c2) # (B, 16, 52, 44)

        c3 = self.conv3_1(p2) # (B, 32, 52, 44)
        c3 = self.conv3_2(c3) # (B, 32, 52, 44)
        p3 = self.pooling(c3) # (B, 32, 26, 22)

        c4 = self.conv4_1(p3) # (B, 64, 26, 22)
        c4 = self.conv4_2(c4) # (B, 64, 26, 22)
        p4 = self.pooling(c4) # (B, 64, 13, 11)

        c5 = self.conv5_1(p4)  # (B, 128, 13, 11)
        out = self.conv5_2(c5)  # (B, 256, 13, 11)

        return out, c4, c3, c2, c1


class IEM(nn.Module):
    # Identity Extracting Module
    def __init__(self):
        super().__init__()
        self.pooling0 = nn.MaxPool2d(kernel_size=16)
        self.pooling1 = nn.MaxPool2d(kernel_size=8)
        self.pooling2 = nn.MaxPool2d(kernel_size=4)
        self.pooling3 = nn.MaxPool2d(kernel_size=2)

        self.conv0_1 = CNNencoder(8, 4)
        self.conv0_2 = CNNencoder(4, 8)
        self.conv1_1 = CNNencoder(16, 8)
        self.conv1_2 = CNNencoder(8, 16)
        self.conv2_1 = CNNencoder(32, 16)
        self.conv2_2 = CNNencoder(16, 32)
        self.conv3_1 = CNNencoder(64, 32)
        self.conv3_2 = CNNencoder(32, 64)
        self.conv4_1 = CNNencoder(256, 128)
        self.conv4_2 = CNNencoder(128, 256)

        self.predictor = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, out, c4, c3, c2, c1):

        c1 = self.conv0_1(c1)
        c1 = self.conv0_2(c1)
        c2 = self.conv1_1(c2)
        c2 = self.conv1_2(c2)
        c3 = self.conv2_1(c3)
        c3 = self.conv2_2(c3)
        c4 = self.conv3_1(c4)
        c4 = self.conv3_2(c4)
        out = self.conv4_1(out)
        out = self.conv4_2(out)

        d_c1 = self.pooling0(c1)
        d_c2 = self.pooling1(c2)
        d_c3 = self.pooling2(c3)
        d_c4 = self.pooling3(c4)

        z = torch.cat((d_c1, d_c2, d_c3, d_c4, out), 1)
        z = torch.mean(z, dim=1, keepdim=True)

        p = self.predictor(z)

        return out, c4, c3, c2, c1, p, z




class Age_predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.AvgPool2d((13, 11)),
            nn.Dropout(0.5),
            nn.Conv2d(256, 33, padding=0, kernel_size=1)
        )

    def forward(self, x):
        out = list()
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        out.append(x)

        return out




class MappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        layers.append((EqualizedLinear(33, 512)))
        layers.append(nn.LeakyReLU(0.2))
        n_mlp = 7
        for i in range(n_mlp):
            layers.append(EqualizedLinear(512, 512))
            layers.append(nn.LeakyReLU(0.2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        age_onehot = age_to_onehot(x)  # scalar to (1, 33) vector
        x = F.normalize(age_onehot.float().cuda(), dim=1)

        return self.net(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = Bottle_CBN(256, 128)
        self.conv1_2 = Skip_CBN(64)
        self.conv2 = CNNencoder(128+64, 64)
        self.conv2_2 = Skip_CBN(32)
        self.conv3 = CNNencoder(64+32, 32)
        self.conv3_2 = Skip_CBN(16)
        self.conv4 = CNNencoder(32+16, 16)
        self.conv4_2 = Skip_CBN(8)
        self.conv5 = CNNencoder(16+8, 4)
        self.out = nn.Sequential(
            spectral_norm(nn.Conv2d(4, 1, kernel_size=1, stride=1, bias=False)))


    def forward(self, x, c4, c3, c2, c1, style):

        u0 = self.conv1(x, style)  # (B, 256, 13, 11)
        u1 = nn.Upsample(scale_factor=2, mode='bilinear').cuda()(u0)  # (B, 256, 26, 22)
        u1 = self.conv1_2(u1, c4, style)  # (B, (256+128), 26, 22)

        u1 = self.conv2(u1) # (B, 128, 26, 22)
        u2 = nn.Upsample(scale_factor=2, mode='bilinear').cuda()(u1)  # (B, 128, 52, 44)
        u2 = self.conv2_2(u2, c3, style)  # (B, (128+64), 52, 44)

        u2 = self.conv3(u2)  # (B, 64, 52, 44)
        u3 = nn.Upsample(scale_factor=2, mode='bilinear').cuda()(u2)  # (B, 64, 104, 88)
        u3 = self.conv3_2(u3, c2, style)  # (B, (64+32), 104, 88)

        u3 = self.conv4(u3)  # (B, 32, 104, 88)
        u4 = nn.Upsample(scale_factor=2, mode='bilinear').cuda()(u3)  # (B, 32, 208, 176)
        u4 = self.conv4_2(u4, c1, style)  # (B, (32+16), 208, 176)

        u4 = self.conv5(u4)  # (B, 8, 208, 176)
        out = self.out(u4)  # (B, 1, 208, 176)

        return out


class Bottle_CBN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.model = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False))
        self.cbn = ConditionalBatchNorm(out_channel)
        self.lrelu = nn.PReLU()

    def forward(self, x, style):

        b1 = self.model(x)
        b1 = self.cbn(b1, style)
        out = self.lrelu(b1)

        return out


class Skip_CBN(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.model1 = spectral_norm(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False))
        self.cbn1 = ConditionalBatchNorm(channel)
        self.lrelu1 = nn.PReLU()

        self.model2 = spectral_norm(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False))
        self.cbn2 = ConditionalBatchNorm(channel)
        self.lrelu2 = nn.PReLU()

    def forward(self, x, skip_x, style):

        s1 = self.model1(skip_x)
        s1 = self.cbn1(s1, style)
        s1 = self.lrelu1(s1)

        s1 = self.model2(s1)
        s1 = self.cbn2(s1, style)
        s1 = self.lrelu2(s1)

        out = torch.cat((x, s1), 1)

        return out




class Conv_CBN_Dis(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()

        self.conv = spectral_norm(nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1, bias=False))
        self.cbn = ConditionalBatchNorm(out_c)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x, style):
        out = self.conv(x)
        out = self.cbn(out, style)
        out = self.lrelu(out)

        return out



class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)



def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module



class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)




class ConditionalBatchNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channel)
        self.style = EqualLinear(512, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out




# PatchGAN Discriminator with Style Transfer
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = Conv_CBN_Dis(1, 32, 4, 2)
        self.conv2 = Conv_BN_Dis(32, 64, 4, 2)
        self.conv3 = Conv_BN_Dis(64, 128, 4, 2)
        self.conv4 = Conv_BN_Dis(128, 256, 4, 2)
        self.out = nn.Sequential(spectral_norm(nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)))

    def forward(self, x, style):

        x1 = self.conv1(x, style) # (B, 64, 45, 54, 45)
        x2 = self.conv2(x1) # (B, 128, 22, 27, 22)
        x3 = self.conv3(x2) # (B, 256, 11, 13, 11)
        x4 = self.conv4(x3) # (B, 512, 5, 6, 5)
        out = self.out(x4) # (B, 1, 4, 5, 4)

        return out


class Conv_BN_Dis(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()

        self.conv = spectral_norm(nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1, bias=False))
        self.bn = nn.BatchNorm2d(out_c)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out