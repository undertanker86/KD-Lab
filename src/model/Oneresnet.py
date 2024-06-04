from resnet import BasicBlock, Bottleneck, conv3x3, conv1x1 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional



class OneResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(OneResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,64, kernel_size=3,  padding=1, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block,64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        fix_inplanes = self.in_planes
        self.layer3_1 = self._make_layer(block, 256, layers[2], stride=2)
        fix_inplanes = self.in_planes
        self.layer3_2 = self._make_layer(block, 256, layers[2], stride=2)
        fix_inplanes = self.in_planes
        self.layer3_3 = self._make_layer(block, 256, layers[2], stride=2)


        self.classfier1_1=nn.Linear(64 * block.expansion, num_classes)
        self.control_v1 = nn.Linear(fix_inplanes, 3)
        self.bn_v1 = nn.BatchNorm2d(16)

        self.avgpool = nn.AvgPool2d((1, 1))

        self.avgpool_c = nn.AvgPool2d(64)

        self.classfier3_1=nn.Linear(256 * block.expansion, num_classes)
        self.classfier3_2=nn.Linear(256 * block.expansion,num_classes)
        self.classfier3_3=nn.Linear(256 * block.expansion, num_classes)




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                self.control_v1 = nn.Linear(fix_inplanes, 3)



        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    
    def forward(self, x):

        # all branches share lower level layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        # x_c is used to give weight to each branch 
        # x_c = self.avgpool_c(x)
        print(x.size()[2:])
        x_c = F.avg_pool2d(x, kernel_size=x.size()[2:])  # Apply average pooling to x
        # x_c = x_c.view(x_c.size(0), -1)  # Flatten the pooled output
        x_c = x_c.view(x_c.size(0), -1)
        print(x_c.shape)
        x_c=self.control_v1(x_c)
        x_c=self.bn_v1(x_c)
        x_c=F.relu(x_c)
        x_c = F.softmax(x_c,dim=1)

        # high level layer is used to make branches
        x_3_1 = self.layer3_1(x)  # 8x8
        x_3_2 = self.layer3_2(x)
        x_3_3 = self.layer3_3(x)


        x_3_1 = self.avgpool(x_3_1)
        x_3_1 = x_3_1.view(x_3_1.size(0), -1)
        x_3_2 = self.avgpool(x_3_2)
        x_3_2 = x_3_2.view(x_3_2.size(0), -1)
        x_3_3 = self.avgpool(x_3_3)
        x_3_3 = x_3_3.view(x_3_3.size(0), -1)

        # Each branch output is weighted by x_c to make teacher ensemble logit
        x_3_1 = self.classfier3_1(x_3_1)
        x_3_2 = self.classfier3_2(x_3_2)
        x_3_3 = self.classfier3_3(x_3_3)
        x_c_1=x_c[:,0].repeat(x_3_1.size()[1], 1).transpose(0,1)
        x_c_2=x_c[:,1].repeat(x_3_1.size()[1], 1).transpose(0,1)
        x_c_3=x_c[:,2].repeat(x_3_1.size()[1], 1).transpose(0,1)
        x_m=x_c_1*x_3_1+x_c_2*x_3_2+x_c_3*x_3_3
        return x_3_1,x_3_2,x_3_3,x_m
    

def one_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OneResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    model = one_resnet18()
    y = model(x)
    print(y.shape)
    print(model)