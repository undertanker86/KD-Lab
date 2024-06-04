import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
  def __init__(self, in_channels_list, reduced_dim=None):
    super(AttentionFusion, self).__init__()
    self.in_channels_list = in_channels_list  # List of input channels from different layers
    self.reduced_dim = reduced_dim  # Optional channel reduction dimension

    # Optional 1x1 convolutions for channel reduction (if needed)
    self.conv1x1_list = nn.ModuleList([nn.Conv2d(in_ch, reduced_dim, kernel_size=1) if reduced_dim else nn.Identity() for in_ch in in_channels_list])

    # Attention modules (replace with your preferred attention implementation)
    self.se_blocks = nn.ModuleList([nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(reduced_dim, reduced_dim // 4, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(reduced_dim // 4, reduced_dim, kernel_size=1),
        nn.Sigmoid()
    ) for _ in in_channels_list])

  def forward(self, x1, x2, x3):
    # Pass through optional 1x1 convolutions for channel reduction
    x1_reduced = self.conv1x1_list[0](x1) if self.reduced_dim else x1
    x2_reduced = self.conv1x1_list[1](x2) if self.reduced_dim else x2
    x3_reduced = self.conv1x1_list[2](x3) if self.reduced_dim else x3

    # Apply attention mechanism (replace with your chosen attention module)
    a1 = self.se_blocks[0](x1_reduced)
    a2 = self.se_blocks[1](x2_reduced)
    a3 = self.se_blocks[2](x3_reduced)

    # Apply attention weights
    x1_refined = x1_reduced * a1
    x2_refined = x2_reduced * a2
    x3_refined = x3_reduced * a3

    # Feature fusion (concatenation)
    combined_features = torch.cat([x1_refined, x2_refined, x3_refined], dim=1)

    # Final 1x1 convolution to adjust channels
    output = F.conv2d(combined_features, self.in_channels_list[-1], kernel_size=1)

    return output

# Example usage (assuming in_channels_list is known)
attention_fusion = AttentionFusion(in_channels_list)
output = attention_fusion(layer1_features, layer2_features, layer3_features)