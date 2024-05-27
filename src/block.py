import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/changzy00/pytorch-attention/blob/master/cnns/mobilenetv1.py
# Origin mobilenetv1 as v2 foward use inverted residual block
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
    def __init__(self, in_channels, out_channels ,kernel_size=3, stride=2):
        super(AttentionModule, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # Note: Not sure if the paper use maxpool or conv1x1 here also have not test comment code
        # self.downsample = conv1x1(in_channels, in_channels, stride)
        self.pwconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        residual = x
        out = self.dwconv(x)
        out = self.downsample(out)
        out = self.pwconv(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.upsample(out)
        # Dot product in the paper why though
        out = out @ residual 
        return out
    

    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ShallowClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, depth):
        super().__init__()

        for i in range(depth):
            setattr(self, f"layer{i+1}", self.make_layers(in_channels, in_channels))
            in_channels = in_channels
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
    
    def make_layers(self, in_channels, out_channels, kernel_size=3, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.depth):
            layer = getattr(self, f"layer{i+1}")
            x = layer(x)
            x += x  # Add skip connection here
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # replacing Flatten and Linear with 2 Conv2d  for MLP can have same result with lower parameter
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
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
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
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
    
if __name__ == "__main__":
    x = torch.randn(2, 64, 56, 56)
    attn = CBAM(64)
    y = attn(x)
    print(y.shape)