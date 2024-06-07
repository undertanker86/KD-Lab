import torch
from torch import nn
import torchvision.models as models
from torch.nn import functional as F
import sys
sys.path.append('src')
from customblock import ChannelAttention, SpatialAttention, CBAM, DepthwiseSeparableConv2d

class TripleAuxResNet(nn.Module):
    def __init__(self, resnet_model='resnet18',num_classes=100, pretrained=True):
        super(TripleAuxResNet, self).__init__()
        if resnet_model == 'resnet18':
            self.pretrained_model = models.resnet18(pretrained=pretrained)
            self.expansion = 1
        elif resnet_model == 'resnet50':
            self.pretrained_model = models.resnet50(pretrained=pretrained)
            self.expansion = 4
        # Freeze the pre-trained layers
        # for param in self.pretrained_model.parameters():
        #     param.requires_grad = False
        self.pretrained_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.pretrained_model.maxpool = nn.Identity()
        # Modify layer1 to create an auxiliary branch
        self.layer1_aux = nn.Sequential(
            self.pretrained_model._modules['layer1'][0],
            CBAM(64*self.expansion),
            DepthwiseSeparableConv2d(64*self.expansion, 128*self.expansion),
            DepthwiseSeparableConv2d(128*self.expansion, 256*self.expansion),
            DepthwiseSeparableConv2d(256*self.expansion, 512*self.expansion),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier1_aux = nn.Linear(512*self.expansion, num_classes)

        # Modify layer2 to create an auxiliary branch
        self.layer2_aux = nn.Sequential(
            self.pretrained_model._modules['layer2'][0],
            CBAM(128*self.expansion),
            DepthwiseSeparableConv2d(128*self.expansion, 256*self.expansion),
            DepthwiseSeparableConv2d(256*self.expansion, 512*self.expansion),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier2_aux = nn.Linear(512*self.expansion, num_classes)

        # Modify layer3 to create an auxiliary branch (similar to layer1 and layer2)
        self.layer3_aux = nn.Sequential(
            self.pretrained_model._modules['layer3'][0],
            CBAM(256*self.expansion),
            DepthwiseSeparableConv2d(256*self.expansion, 512*self.expansion),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier3_aux = nn.Linear(512*self.expansion, num_classes)

        # Replace the original layers with modified versions
        # self.pretrained_model._modules['layer1'] = self.layer1_aux
        # self.pretrained_model._modules['layer2'] = self.layer2_aux
        # self.pretrained_model._modules['layer3'] = self.layer3_aux

        # Final classification head (same as original ResNet)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*self.expansion , num_classes)

    def forward(self, x):
        # Pass through pre-trained layers up to layer0 (conv1)
        out = self.pretrained_model.conv1(x)
        out = self.pretrained_model.bn1(out)
        out = self.pretrained_model.relu(out)
        out = self.pretrained_model.maxpool(out)
       
        # Auxiliary branch for layer1
        aux1 = self.layer1_aux(out)
        aux1 = aux1.view(aux1.size(0), -1)
        out1 = self.classifier1_aux(aux1)

        # Main branch through layer1
        out = self.pretrained_model.layer1(out)
        
        # Auxiliary branch for layer2
        aux2 = self.layer2_aux(out)
        aux2 = aux2.view(aux2.size(0), -1)
        out2 = self.classifier2_aux(aux2)

        # Main branch through layer2
        out = self.pretrained_model.layer2(out)

        # Auxiliary branch for layer3
        aux3 = self.layer3_aux(out)
        aux3 = aux3.view(aux3.size(0), -1)
        out3 = self.classifier3_aux(aux3)

        # Main branch through layer3 and beyond
        out = self.pretrained_model.layer3(out)
        out = self.pretrained_model.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out4 = self.fc(out)

        return [out1, out2, out3, out4]
    

if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    model = TripleAuxResNet()
    y = model(x)
    print(y[0].shape)