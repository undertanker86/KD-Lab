from torch import nn
import torch
from custom_block import *


class MobileNetv1(nn.Module):
    def __init__(self, width_multiplier=1, num_classes=1000):
        super().__init__()
        alpha = width_multiplier
        self.stem = nn.Sequential(
            BasicConv2d(in_channels=3, out_channels=int(32 * alpha), stride=2),
            DepthwiseSeparableConv2d(in_channels=int(32 * alpha), out_channels=int(64 * alpha))
        )
        self.layer1 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(64 * alpha), out_channels=int(128 * alpha), stride=2),
            DepthwiseSeparableConv2d(in_channels=int(128 * alpha), out_channels=int(128 * alpha))
        )
        self.layer2 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(128 * alpha), out_channels=int(256 * alpha), stride=2),
            DepthwiseSeparableConv2d(in_channels=int(256 * alpha), out_channels=int(256 * alpha))
        )
        self.layer3 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(256 * alpha), out_channels=int(512 * alpha), stride=2),
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha)),
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha)),
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha)),
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha)),
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(512 * alpha)),
        )
        self.layer4 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=int(512 * alpha), out_channels=int(1024 * alpha), stride=2),
            DepthwiseSeparableConv2d(in_channels=int(1024 * alpha), out_channels=int(1024 * alpha))
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(int(1024 * alpha), num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = MobileNetv1()
    y = model(x)
    print(y.shape)