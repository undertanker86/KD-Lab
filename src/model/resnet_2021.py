# This file is for paper 2022 toward compact neural network
import sys
sys.path.append('src')

from customblock import CBAM, DepthwiseSeparableConv2d
from resnet import BasicBlock, Bottleneck,PreActBlock, conv3x3, conv1x1 
import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class ResNet_Final_Auxiliary_Classifer(nn.Module):
    def __init__(self, block, num_classes):
        super(ResNet_Final_Auxiliary_Classifer, self).__init__()
        self.conv = conv1x1(512 * block.expansion * 4, 512 * block.expansion)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def forward(self, x):
        sum_fea = torch.cat(x, dim=1)
        out = self.conv(sum_fea)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    
class Auxiliary_Classifier(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Auxiliary_Classifier, self).__init__()
        
        layers = [1, 1, 1, 1]
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64 * block.expansion
        self.block_extractor1 = nn.Sequential(*[self._make_layer(block, 128, layers[1], stride=2),
                                                self._make_layer(block, 256, layers[2], stride=2),
                                                self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = 128 * block.expansion
        self.block_extractor2 = nn.Sequential(*[self._make_layer(block, 256, layers[2], stride=2),
                                                self._make_layer(block, 512, layers[3], stride=2)])

        self.inplanes = 256 * block.expansion
        self.block_extractor3 = nn.Sequential(*[self._make_layer(block, 512, layers[3], stride=2)])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        aux_logits = []
        aux_feats = []
        for i in range(len(x)):
            idx = i + 1
            out = getattr(self, 'block_extractor'+str(idx))(x[i])
            aux_feats.append(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            
            out = getattr(self, 'fc'+str(idx))(out)
            aux_logits.append(out)
            
        return aux_logits, aux_feats


class CifarResNet2021(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck, PreActBlock]],
        layers: List[int],
        branch_block: Type[Union[BasicBlock, Bottleneck, PreActBlock, DepthwiseSeparableConv2d,CBAM]]= None,
        branch_layers: List[int]= [],#
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dataset_type: str = 'cifar',
        
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            block (Type[Union[BasicBlock, Bottleneck, PreActBlock]]): The type of residual block to use.
            layers (List[int]): The number of layers in each residual block.
            branch_block (Type[Union[BasicBlock, Bottleneck, PreActBlock, DepthwiseSeparableConv2d,CBAM]]): The type of residual block to use in the branching layers.
            branch_layers (List[int], optional): The number of layers in each branching block. Defaults to [].
            num_classes (int, optional): The number of output classes. Defaults to 1000.
            zero_init_residual (bool, optional): If True, initializes the last batch normalization layer of each residual block with zeros. Defaults to False.
            groups (int, optional): The number of groups for grouped convolutions. Defaults to 1.
            width_per_group (int, optional): The number of channels per group for grouped convolutions. Defaults to 64.
            replace_stride_with_dilation (Optional[List[bool]], optional): If provided, replaces the 2x2 stride with dilated convolutions in the specified positions. Defaults to None.
            norm_layer (Optional[Callable[..., nn.Module]], optional): The normalization layer to use. Defaults to None.
            dataset_type (str, optional): The type of dataset being used. Defaults to 'cifar'.

        Raises:
            ValueError: If replace_stride_with_dilation is not None and does not have 3 elements.
            NotImplementedError: If dataset_type is not 'imagenet' or 'cifar'.

        Returns:
            None
        """
        super().__init__()
        self.branch_layers = branch_layers
        self.branch_block = branch_block
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dataset_type = dataset_type
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
        
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )   

        
        self.groups = groups
        self.base_width = width_per_group
        if self.dataset_type == 'imagenet':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif self.dataset_type == 'cifar':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            raise NotImplementedError
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # inplanes_head3 = self.inplanes # 64*block.expansion
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])

        # inplanes_head2 = self.inplanes # 128*block.expansion
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # inplanes_head1 = self.inplanes # 256*block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        

        self.net_channel = [64*block.expansion, 128*block.expansion, 256*block.expansion, 512*block.expansion]


        if len(self.branch_layers)!= 0:
            # check if branch_block is CBAM
            if self.branch_block == CBAM:
                self.branch_layer1 = branch_block(64)
                self.branch_layer2 = branch_block(128)
                self.branch_layer3 = branch_block(256)
            elif self.branch_block == DepthwiseSeparableConv2d:
                self.branch_layer1 = branch_block(64)
                self.branch_layer2 = branch_block(128)
                self.branch_layer3 = branch_block(256)
            elif self.branch_block == None:
                self.branch_layer1 = None
                self.branch_layer2 = None
                self.branch_layer3 = None
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor,y=None) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        out1 = x
        x = self.layer2(x)
        out2 = x
        x = self.layer3(x)
        out3 = x
        x = self.layer4(x)
        out4 = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logit = self.fc(x)    
        if len(self.branch_layers)!= 0:
            f1 = self.branch_layer1(out1)
            
            f2 = self.branch_layer2(out2)
            f3 = self.branch_layer3(out3)
            
            
        
            return [logit],[f1,f2,f3,out4]
        
        return logit
    

class ResnetAuxiliary(nn.Module):
    def __init__(self,  block, layers, branch_block, branch_layers, aux_block,num_classes=1000, zero_init_residual=False, ensemble=True):
        super().__init__()
        self.backbone = CifarResNet2021(block,layers, branch_block, branch_layers,num_classes=num_classes, zero_init_residual=zero_init_residual)
        self.aux_classifier = Auxiliary_Classifier(aux_block,layers,num_classes)
        self.final_classifier = ResNet_Final_Auxiliary_Classifer(aux_block,num_classes)
        self.ensemble = ensemble

    def forward(self, x):
        logit, feature = self.backbone(x)
        aux_logit = self.aux_classifier(feature[:-1])
        aux_logit.append(logit)
        if self.ensemble:
            ensemble_feature = torch.stack(feature).mean(dim=0)
            ensemble_logit = self.final_classifier(feature)
            return ensemble_logit, ensemble_feature 
        return aux_logit, feature


def resnet18(pretrained: bool = False, **kwargs: Any) -> CifarResNet2021:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CifarResNet2021(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet18_cbam(pretrained: bool = False, **kwargs: Any) -> CifarResNet2021:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResnetAuxiliary(BasicBlock, [2, 2, 2, 2], CBAM, [1, 1, 1, 1], BasicBlock, **kwargs)
    return model



if __name__ == '__main__':
    x = torch.randn(1, 3, 32, 32)
    model = resnet18_cbam(pretrained=False)
    o,f = model(x)
    print(len(o),len(x))
    # import torchviz
    # torchviz.make_dot(y[0].mean(), params=dict(list(model.named_parameters()))).render("resnet18_cbam", format="png")
    # for i, j in enumerate(y):
    #     print(i, j.shape)
