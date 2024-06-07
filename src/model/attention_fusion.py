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
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class FeatureFusionWithAttention(nn.Module):
    def __init__(self):
        super(FeatureFusionWithAttention, self).__init__()
        self.resnet50 = ResNet50WithFeatures(pretrained=True)
        
        self.attention1 = ChannelAttention(in_channels=256)  # For layer1 output
        self.attention2 = ChannelAttention(in_channels=512)  # For layer2 output
        self.attention3 = ChannelAttention(in_channels=1024) # For layer3 output

        self.conv1 = nn.Conv2d(256, 1024, kernel_size=1) # Align channels
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=1)
        
        self.self_attention = MultiheadAttention(embed_dim=1024, num_heads=8)
        self.fc = nn.Linear(1024, 1000)  # Assuming 1000 classes for classification

    def forward(self, x):
        out1, out2, out3 = self.resnet50(x)
        
        att1 = self.attention1(out1) * out1
        att2 = self.attention2(out2) * out2
        att3 = self.attention3(out3) * out3

        att1 = self.conv1(att1)
        att2 = self.conv2(att2)

        # Flatten the spatial dimensions for self-attention
        batch_size = att1.size(0)
        att1 = att1.view(batch_size, 1024, -1).permute(2, 0, 1)
        att2 = att2.view(batch_size, 1024, -1).permute(2, 0, 1)
        att3 = att3.view(batch_size, 1024, -1).permute(2, 0, 1)

        # Concatenate along the sequence length dimension
        fused = torch.cat((att1, att2, att3), dim=0)
        
        # Apply self-attention
        att_output, _ = self.self_attention(fused, fused, fused)
        
        # Reshape back to the original dimension
        att_output = att_output.permute(1, 2, 0).view(batch_size, 1024, *out3.shape[2:])
        
        # Global average pooling
        fused = F.adaptive_avg_pool2d(att_output, (1, 1)).view(batch_size, -1)
        out = self.fc(fused)
        
        return out
