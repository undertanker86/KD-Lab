import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================== Channel Attention Map ===================================
# tìm ra channel nào đóng góp sự quan trọng nhất
# output sẽ là một refined feature map
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CAM(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            Flatten(),  # (b x c)
            nn.Linear(in_channels, in_channels //
                      reduction_ratio),  # (b x (c/r))
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),  # (b x (c/r))
            nn.Linear(in_channels // reduction_ratio, in_channels),  # (b x c)
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.pool_types = pool_types

    def forward(self, x):
        # x: (b x c x h x w)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # (b x c x 1 x 1)
                channel_att_raw = self.mlp(avg_pool) # (b x c)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))) # (b x c x 1 x 1)
                channel_att_raw = self.mlp(max_pool) # (b x c)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw # (b x c)
            else:
                channel_att_sum = channel_att_sum + channel_att_raw # (b x c)

        # print(channel_att_sum.unsqueeze(2).unsqueeze(3).shape) # (b x c x 1 x 1)
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x) # (b x c x h x w)
        return x * scale

"""
if __name__ == '__main__':
    x = torch.rand(3, 4, 32, 32)
    y = torch.rand(3, 4, 1, 1)
    print(y)
    print(y.expand_as(x))
    # cam = ChannelGate(48)
    # out = cam(x)
    # print(out.shape)
    # fl = Flatten()
    # avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)),
    #                         stride=(x.size(2), x.size(3)))
    # print(avg_pool)
    # out = fl(avg_pool)
    # print(out)
"""


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], is_spatial=True):
        super(CBAM, self).__init__()
        self.cam = CAM(in_channels, reduction_ratio, pool_types)
        self.is_spatial = is_spatial
        if is_spatial:
            self.sam = SAM()

    def forward(self, x):
        out = self.cam(x)
        if self.is_spatial:
            out = self.sam(out)
        return out



# ================================== Spatial Attention Map ===================================
# tìm ra pixel nào đóng góp sự quan trọng nhất
# output sẽ là một refined feature map
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels, affine=True, momentum=0.99, eps=1e-3) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPooling(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        self.pool = ChannelPooling()
        self.conv = ConvLayer(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2, relu=False)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)
        scale = torch.sigmoid(out)
        return scale * x


import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, affine=True, momentum=0.99, eps=1e-3),
        nn.ReLU(inplace=True)
    )

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, dilation=dilation, groups=in_channels,
                                   bias=bias)
        self.bnd = nn.BatchNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride, 0, 1, 1, bias=bias)
        self.bnp = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.relu(self.bnd(out))
        out = self.pointwise(out)
        out = self.relu(self.bnp(out))
        return out

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, keep_dim=False):
        super(Block, self).__init__()

        self.keep_dim = keep_dim
        stride_sep_conv2 = 2
        if keep_dim:
            stride_sep_conv2 = 1
        self.conv = conv3x3(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sep_conv1 = SeparableConv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1)
        self.sep_conv2 = SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=stride_sep_conv2, bias=False, padding=1)
        self.cbam = CBAM(out_channels)
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2) if not keep_dim else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv(x)
        if not self.keep_dim:
            residual = self.maxp(residual)

        out = self.sep_conv1(x)
        out = self.sep_conv2(out)
        out = self.cbam(out)
        out += residual
        # out = self.relu(out)
        return out
# class

class Model(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 1
        self.conv1 = conv3x3(in_channels, 8)
        self.conv2 = conv3x3(8, 8)
        self.cbam = CBAM(8,8)

        # 2
        self.block1 = Block(8, 16, keep_dim=True)

        # 3
        self.block2 = Block(16, 32, keep_dim=True)

        # 4
        self.block3 = Block(32, 64, keep_dim=False)

        # 5
        self.block4 = Block(64, 128, keep_dim=False)

        # last conv to down to num_classes
        self.last_conv = conv3x3(128, num_classes)

        # global avg-p
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Initial convolutions
        out = self.conv1(x)
        #print("After conv1", out.size())
        out = self.conv2(out)
        #print("After conv2", out.size())
        out = self.cbam(out)
        #print("After CBAM", out.size())

        # Block processing
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        #print("After blocks", out.size())

        # Final convolution and pooling
        out = self.last_conv(out)
        #print("After last_conv", out.size())
        out = self.avgp(out)
        #print("After avgp", out.size())
        out = out.view((out.shape[0], -1))

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv_1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

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



class ResNet(nn.Module):

    def __init__(self, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

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

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # FGW models
        """
        self.fgw1 = Model(64, num_classes)
        self.fgw2 = Model(128, num_classes)
        self.fgw3 = Model(256, num_classes)
        self.fgw4 = Model(512, num_classes)
        """

        self.block1_fgw = nn.Sequential(Block(64, 128), Block(128, 256), Block(256, 512))
        self.block2_fgw = nn.Sequential(Block(128, 256), Block(256, 512))
        self.block3_fgw = Block(256, 512)
        self.block4_fgw = Block(512, 512, keep_dim=False)

        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(512, num_classes)
        self.fc4 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    """
    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv_1x1(self.inplanes, planes, stride),
                self._norm_layer(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)
    """

    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv_1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                norm_layer(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        out1 = x # 64*h*w
        out1 = self.block1_fgw(out1) #64*h*w
        out1 = self.avgpool(out1)
        out1 = torch.flatten(out1, 1)# 512*1*1
    
 
        logit1 = self.fc1(out1)

        x = self.layer2(x)
        out2 = self.block2_fgw(x)
        out2 = self.avgpool(out2)
        out2 = torch.flatten(out2, 1)
        logit2 = self.fc2(out2)

        x = self.layer3(x)
        out3 = self.block3_fgw(x)
        out3 = self.avgpool(out3)
        out3 = torch.flatten(out3, 1)
        logit3 = self.fc3(out3)

        x = self.layer4(x)
#512*h/4*w/4
        # out4 = self.block4_fgw(x)
        out4 = self.avgpool(x)# 512*1*1
        out4 = torch.flatten(out4, 1)
        logit4 = self.fc4(out4)

        return [out4, out3, out2, out1], [logit4, logit3, logit2, logit1]

def _resnet(arch, layers, pretrained, progress, **kwargs):
    model = ResNet(layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

if __name__ == '__main__':
    x = torch.rand(1,3,32, 32)
    net = resnet18()
    # net = Block(64, 128, keep_dim=False)
    # net = nn.Sequential(Block(64, 128), Block(128, 256, keep_dim=False), Block(256, 512, keep_dim=False))
    out = net(x)
    print(out[1][0].shape)
    # # import torch
    # from torchvision.models import resnet18
    # model = resnet18(pretrained=False)
    # input_tensor = torch.randn(1, 256,8, 8)  # Assuming input size is 224x224
    # output = model.layer4(input_tensor)
    # print(output.shape)