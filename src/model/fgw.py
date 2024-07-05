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

class FGW(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FGW, self).__init__()
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

class FGWLinear(nn.Module):
    def __init__(self, in_channels, num_classes,dropout=0.5):
        super(FGWLinear, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, 8)
        self.conv2 = conv3x3(8, 8)
        self.cbam = CBAM(8,8)

        self.block1 = Block(8, 16, keep_dim=True)

        self.block2 = Block(16, 32, keep_dim=True)

        self.block3 = Block(32, 64, keep_dim=False)

        self.block4 = Block(64, 128, keep_dim=False)
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))
        # self.last_conv = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # Initial convolutions
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.cbam(out)

        # Block processing
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        # Final convolution and pooling
        # out = self.last_conv(out)
        out = self.avgp(out)
        
        out = self.dropout(out)
        out = out.view((out.shape[0], -1))
  
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    model = FGWLinear(3, 10)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)