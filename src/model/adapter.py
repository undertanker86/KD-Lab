import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )
    def forward(self, x):
        return self.op(x)
    

class CustomHead(nn.Module):
    def __init__(self, in_planes, num_classes, pool_size,features=False):
        super(CustomHead, self).__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.fc = nn.Linear(in_planes * pool_size[0] * pool_size[1], num_classes)

    def forward(self, x):
        x = self.pool(x)
        fea = x.view(x.size(0), -1)
        x = self.fc(fea)
        if self.features:
            return x, fea
        return x
    

class AdapterResnet1(nn.Module):
    def __init__(self,  block, attention, num_classes=100, pool_size=(4,4),expand=[64,128,256,512],features=False):
        super(AdapterResnet1, self).__init__()
        self.features = features
        self.expand = expand
        if attention is not None:
            self.attention = attention(self.expand[0])
        else:
            self.attention = nn.Identity()
        # 1*64*56*56->1*512*7*7
        self.scalenet = nn.Sequential(
            block(self.expand[0], self.expand[1]),
            block(self.expand[1], self.expand[2]),
            block(self.expand[2], self.expand[3]),
        )
        self.head = CustomHead(self.expand[3], num_classes, pool_size=pool_size,features=features)

    def forward(self, x):
        if self.attention is not None:
            att = self.attention(x)
            fea = att * x
        else:
            fea = x
        fea = self.scalenet(fea)
        out,fea = self.head(fea)
        if self.features:
            return out, fea
        return out
    
class AdapterResnet2(nn.Module):
    def __init__(self, block, attention, num_classes=100, pool_size=(4,4),expand=[128,256,512],features=False):
        super(AdapterResnet2, self).__init__()
        self.features = features
        self.expand = expand
        if attention is not None:
            self.attention = attention(self.expand[0])
        else:
            self.attention = nn.Identity()
        self.scalenet = nn.Sequential(
            block(self.expand[0], self.expand[1]),
            block(self.expand[1], self.expand[2]),
        )
        self.head = CustomHead(self.expand[2], num_classes, pool_size=pool_size,features=features)

    def forward(self, x):
        if self.attention is not None:
            att = self.attention(x)
            fea = att * x
        else:
            fea = x
        fea = self.scalenet(fea)
        out,fea = self.head(fea)
        if self.features:
            return out, fea
        return out


class AdapterResnet3(nn.Module):
    def __init__(self, block, attention, num_classes=100, pool_size=(4,4),expand=[256,512],features=False):
        super(AdapterResnet3, self).__init__()
        self.features = features
        self.expand = expand
        if attention is not None:
            self.attention = attention(self.expand[0])
        else:
            self.attention = nn.Identity()
        self.scalenet = nn.Sequential(
            block(self.expand[0], self.expand[1]),
        )
        self.head = CustomHead(self.expand[1], num_classes, pool_size=pool_size,features=features)

    def forward(self, x):
        if self.attention is not None:
            att = self.attention(x)
            fea = att * x
        else:
            fea = x
        fea = self.scalenet(fea)
        out,fea = self.head(fea)
        if self.features:
            return out, fea
        return out
    
