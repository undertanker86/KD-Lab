import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4, num_experts)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, data): 
        x = self.linear1(data)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class MoE_ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(MoE_ResNet18, self).__init__()
        
        # Load pre-trained ResNet-18 model
        self.resnet = timm.create_model(model_name='resnet18', pretrained=False, features_only=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.maxpool = nn.Identity()
        # Remove the fully connected layer
        
        # Create classifiers for each layer
        self.classifier1 = nn.Linear(64, num_classes)
        self.classifier2 = nn.Linear(128, num_classes)
        self.classifier3 = nn.Linear(256, num_classes)
        self.classifier4 = nn.Linear(512, num_classes)
        

        # Gating network
        self.gating_network = GatingNetwork(input_size=(64+128+256+512), num_experts=4)

    
    def forward(self, x):
        # Layer outputs
        x = self.resnet(x)
        layer1_output = x[1]
        layer2_output = x[2]
        layer3_output = x[3]
        layer4_output = x[4]
        
        # Pooling to match classifier input size
        layer1_output = F.adaptive_avg_pool2d(layer1_output, (1, 1)).view(layer1_output.size(0), -1)
        layer2_output = F.adaptive_avg_pool2d(layer2_output, (1, 1)).view(layer2_output.size(0), -1)
        layer3_output = F.adaptive_avg_pool2d(layer3_output, (1, 1)).view(layer3_output.size(0), -1)
        layer4_output = F.adaptive_avg_pool2d(layer4_output, (1, 1)).view(layer4_output.size(0), -1)
        
        # Classifier outputs
        out1 = self.classifier1(layer1_output)
        out2 = self.classifier2(layer2_output)
        out3 = self.classifier3(layer3_output)
        out4 = self.classifier4(layer4_output)
        # print(layer1_output.shape, layer2_output.shape, layer3_output.shape, layer4_output.shape)
        # Concatenate all layer outputs for gating network input but scale to same dimensions
        combined_features = torch.cat((layer1_output, layer2_output, layer3_output, layer4_output), dim=1)

        # Gating network output
        gating_weights = self.gating_network(combined_features)
        
        # Weighted sum of classifier outputs
        expert_outputs = torch.stack((out1, out2, out3, out4), dim=1)
        final_output = torch.sum(expert_outputs * gating_weights.unsqueeze(-1), dim=1)
        
        return out1, out2, out3, out4,final_output 
if __name__ == '__main__':
    # Create the model
    model = MoE_ResNet18(num_classes=10)

    # Example input
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output)
