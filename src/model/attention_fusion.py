import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import MultiheadAttention


class ResNet50WithFeatures(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50WithFeatures, self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        self.layer1 = nn.Sequential(*list(self.resnet50.children())[:5])  # Output of layer1
        self.layer2 = nn.Sequential(*list(self.resnet50.children())[5])   # Output of layer2
        self.layer3 = nn.Sequential(*list(self.resnet50.children())[6])   # Output of layer3

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out1, out2, out3

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
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
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

class FeatureFusionWithAttention(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureFusionWithAttention, self).__init__()
        self.resnet50 = ResNet50WithFeatures(pretrained=pretrained)
        
        self.attention1 = CBAM(256)  # For layer1 output
        self.attention2 = CBAM(512)  # For layer2 output
        self.attention3 = CBAM(1024) # For layer3 output

        self.conv1 = nn.Conv2d(256, 1024, kernel_size=1) # Align channels
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=1)
        
        self.self_attention = MultiheadAttention(embed_dim=1024, num_heads=8)
        self.fc = nn.Linear(1024, 1000)  # Assuming 1000 classes for classification

    def forward(self, x):
        out1, out2, out3 = self.resnet50(x)
        print(out1.shape, out2.shape, out3.shape)
        att1 = self.attention1(out1) * out1
        att2 = self.attention2(out2) * out2
        att3 = self.attention3(out3) * out3

        att1 = self.conv1(att1)
        att2 = self.conv2(att2)

        # Flatten the spatial dimensions for self-attention
        batch_size = att1.size(0)
        spatial_dim = att1.size(2) * att1.size(3)
        att1 = att1.view(batch_size, 1024, -1).permute(2, 0, 1)
        att2 = att2.view(batch_size, 1024, -1).permute(2, 0, 1)
        att3 = att3.view(batch_size, 1024, -1).permute(2, 0, 1)

        # Concatenate along the sequence length dimension
        fused = torch.cat((att1, att2, att3), dim=0)
        
        # Apply self-attention
        att_output, _ = self.self_attention(fused, fused, fused)
        
        # Reshape back to the original dimension
        print(att_output.shape)
        att_output = att_output.permute(1, 2, 0).contiguous()
        att_output = att_output.view(batch_size, 1024, int(spatial_dim**0.5), int(spatial_dim**0.5))
        att_output = att_output.permute(1, 2, 0).view(batch_size, 1024, *out3.shape[2:])
        
        # Global average pooling
        fused = F.adaptive_avg_pool2d(att_output, (1, 1)).view(batch_size, -1)
        out = self.fc(fused)
        
        return out

if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = FeatureFusionWithAttention(pretrained=False)
    y = model(x)
    print(y.shape)

    # x = torch.randn(2, 256, 32, 32)
    # model = CBAM(256)
    # y = model(x)
    # print(y.shape)