import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional

# from https://github.com/changzy00/pytorch-attention/blob/master/cnns/mobilenetv1.py
# Origin mobilenetv1 as v2 foward use inverted residual block
# Make change for this to work with expansion


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pwconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Doing guess work but maybe only using 2 conv without BatchNorm is better
class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(AttentionModule, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # Note: Not sure if the paper use maxpool or conv1x1 here also have not test comment code
        # self.downsample = conv1x1(in_channels, in_channels, stride)
        self.pwconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        residual = x
        out = self.dwconv(x)
        out = self.downsample(out)
        out = self.pwconv(out)
        out = F.interpolate(out, scale_factor=2,
                            mode="bilinear", align_corners=True)
        out = self.upsample(out)
        # Dot product in the paper why though
        out = out @ residual
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward(self, x):
        aux_logits = []
        aux_feats = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, "block_extractor" + str(idx))(x[i])
            aux_feats.append(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)

            out = getattr(self, "fc" + str(idx))(out)
            aux_logits.append(out)

        return aux_logits, aux_feats


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # replacing Flatten and Linear with 2 Conv2d  for MLP can have same result with lower parameter
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module

    As described in https://arxiv.org/pdf/1807.06521"""

    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ModifySpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(
            2, 512, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class ModCBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module

    As described in https://arxiv.org/pdf/1807.06521"""

    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = CBAM(channel, reduction, kernel_size)
        self.sa = ModifySpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


# Example usage:
# Should print torch.Size([1, 512, 14, 14])

if __name__ == "__main__":
    # x = torch.randn(2, 64, 56, 56)
    # model = ChannelAttention(64)
    # out = model(x)
    # print(out)
    # print(out.shape)

    # input_tensor = torch.randn(2, 64, 56, 56)
    # model = ConvolutionalNetwork()
    # output_tensor = model(input_tensor)
    # print(output_tensor.shape)

    # input_tensor = torch.randn(1, 64,56, 56)
    # model = nn.Sequential(
    #     nn.Conv2d(64,512,3,1,1),
    #     CBAM(512),
    #     nn.Conv2d(512,512,3,2,1),
    #     nn.Conv2d(512,512,3,2,1)
    # )
    # output_tensor = model(input_tensor)
    # print(output_tensor.shape)

    # input_tensor = torch.rand(1,64,32,32)
    model = nn.Sequential(
        DepthwiseSeparableConv2d(64*4, 128*4, kernel_size=1, stride=2),
        DepthwiseSeparableConv2d(128*4, 256*4, kernel_size=1, stride=2),
        DepthwiseSeparableConv2d(256*4, 512*4, kernel_size=1, stride=1),
    )
    # output_tensor = model(input_tensor)
    # print(output_tensor.shape)
    # fc = nn.Sequential(
    #     nn.AdaptiveAvgPool2d((1,1)),
    #     nn.Linear(512, 10),
    # )
    # output_tensor = fc(output_tensor)
    # print(output_tensor.shape)

    # print(CBAM == CBAM)
    from utils import cal_param_size, cal_multi_adds
    # model = CBAM(128)
    print(cal_param_size(model))
