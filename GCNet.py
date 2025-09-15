import torch
import torch.nn as nn
import torch.nn.functional as F

class GCBlock(nn.Module):
    def __init__(self, in_channels, ratio=4, name=None):
        super(GCBlock, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.name = name
        
        # Calculate bottleneck channels (ensuring it's at least 1)
        self.bottleneck_channels = max(1, in_channels // ratio)
        
        # Spatial mask
        self.mask_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        
        # Transform layers
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, self.bottleneck_channels, kernel_size=1, bias=False),
            nn.LayerNorm([self.bottleneck_channels, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck_channels, in_channels, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        # Input shape: [B, C, H, W]
        batch_size, channels, height, width = x.size()
        
        # Spatial mask
        mask = self.mask_conv(x)  # [B, 1, H, W]
        mask = mask.view(batch_size, 1, -1)  # [B, 1, H*W]
        mask = F.softmax(mask, dim=-1)  # Softmax for spatial attention weights
        
        # Context vector
        x_flat = x.view(batch_size, channels, -1)  # [B, C, H*W]
        context = torch.matmul(mask, x_flat.transpose(1, 2))  # [B, 1, C]
        context = context.unsqueeze(-1)  # [B, 1, C, 1]
        context = context.transpose(1, 2)  # [B, C, 1, 1]
        
        # Transform context
        transformed_context = self.transform(context)
        
        # Add back to input
        return x + transformed_context