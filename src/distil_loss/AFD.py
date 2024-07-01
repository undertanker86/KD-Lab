import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet18
import timm

class AFD(nn.Module):
    def __init__(self, in_channels, att_f):
        super(AFD, self).__init__()
        mid_channels = int(in_channels * att_f)

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, fm_s, fm_t, eps=1e-6):
        fm_t_pooled = F.adaptive_avg_pool2d(fm_t, 1)
        rho = self.attention(fm_t_pooled)
        
        rho = torch.sigmoid(rho)
        
        rho = rho / torch.sum(rho, dim=1, keepdim=True)
    
        fm_s_norm = torch.norm(fm_s, dim=(2,3), keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm+eps)
        fm_t_norm = torch.norm(fm_t, dim=(2,3), keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm+eps)

        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=(2,3))
        loss = loss.sum(1).mean(0)
        return loss

if __name__ == '__main__':
    class ResNet18WithFeatures(nn.Module):
        def __init__(self):
            super(ResNet18WithFeatures, self).__init__()
            self.backbone = timm.create_model(model_name='resnet18', pretrained=False, features_only=True)
            


        def forward(self, x):
            x = self.backbone(x)

            fm_s1 = x[1]
            fm_t = x[4]

            return fm_s1, fm_t

    # Define student and teacher models
    teacher_model = ResNet18WithFeatures()
    student_model = ResNet18WithFeatures()

    # Instantiate AFD module
    afd_module = AFD(in_channels=512, att_f=0.5)  # in_channels should match the teacher feature map channels

    # Dummy input
    x = torch.randn(1, 3, 224, 224)

    # Forward pass
    student_features, teacher_features = student_model(x)

    # Align student features to match teacher features
    aligned_student_features = F.adaptive_avg_pool2d(student_features, (7, 7))  # Align spatial dimensions

    # Ensure the number of channels matches
    conv_align = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
    aligned_student_features = conv_align(aligned_student_features)

    # Compute AFD loss
    loss = afd_module(aligned_student_features, teacher_features)

    print("AFD Loss:", loss.mean().item())