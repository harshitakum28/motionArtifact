import torch
import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    """
    Hard Swish activation function used in the original Coordinate Attention paper.
    Mathematically: x * ReLU6(x + 3) / 6
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class HSigmoid(nn.Module):
    """
    Hard Sigmoid activation function used in the original Coordinate Attention paper.
    Mathematically: ReLU6(x + 3) / 6
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=True) / 6.0


class ArtifactAwareCoordinateAttention(nn.Module):
    """
    Artifact-Aware Coordinate Attention Module adapted for 3D MRI patches.
    
    This module extends the CVPR 2021 Coordinate Attention methodology to 3D.
    Instead of 2D (H, W) pooling, it performs pooling along the three spatial 
    dimensions (D, H, W) of the MRI patch to capture long-range artifact smearing
    and structural dependencies.
    
    The input feature map has shape (B, C, D, H, W).
    """
    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for the intermediate channel dimension.
        """
        super(ArtifactAwareCoordinateAttention, self).__init__()
        mip = max(8, channels // reduction)

        # 3D pooling operations to capture spatial directions independently
        # Pool (H, W) -> retain D
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        # Pool (D, W) -> retain H
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        # Pool (D, H) -> retain W
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))

        # Shared 1x1x1 convolution
        self.conv1 = nn.Conv3d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = HSwish()

        # Separate 1x1x1 convolutions to generate attention for each axis
        self.conv_d = nn.Conv3d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, channels, kernel_size=1, stride=1, padding=0)

        self.sigmoid = HSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Artifact-Aware Coordinate Attention.
        
        Args:
            x (torch.Tensor): Feature map of shape (B, C, D, H, W)
            
        Returns:
            torch.Tensor: Attention-weighted feature map of shape (B, C, D, H, W)
        """
        identity = x
        
        b, c, d, h, w = x.size()

        # 1. Coordinate Information Embedding
        # Extract direction-aware representations
        x_d = self.pool_d(x)  # (B, C, D, 1, 1)
        x_h = self.pool_h(x)  # (B, C, 1, H, 1)
        x_w = self.pool_w(x)  # (B, C, 1, 1, W)

        # 2. Coordinate Attention Generation
        # Reshape to concatenate along the spatial dimension (we flatten D, H, W to 1D sequences)
        x_d_flat = x_d.view(b, c, 1, 1, d) # Treat as W dim for concatenation
        x_h_flat = x_h.view(b, c, 1, 1, h)
        x_w_flat = x_w.view(b, c, 1, 1, w)

        # Concatenate along the last dimension
        y = torch.cat([x_d_flat, x_h_flat, x_w_flat], dim=4) # (B, C, 1, 1, D + H + W)

        # Apply shared 1x1x1 conv, batch norm, and non-linearity
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) # (B, mip, 1, 1, D + H + W)

        # Split the feature map back into individual spatial components
        y_d_flat, y_h_flat, y_w_flat = torch.split(y, [d, h, w], dim=4)
        
        # Reshape back to original pooled tensor shapes
        y_d = y_d_flat.view(b, -1, d, 1, 1) # (B, mip, D, 1, 1)
        y_h = y_h_flat.view(b, -1, 1, h, 1) # (B, mip, 1, H, 1)
        y_w = y_w_flat.view(b, -1, 1, 1, w) # (B, mip, 1, 1, W)

        # Apply axis-specific 1x1x1 convs and sigmoid gating
        a_d = self.sigmoid(self.conv_d(y_d)) # (B, C, D, 1, 1)
        a_h = self.sigmoid(self.conv_h(y_h)) # (B, C, 1, H, 1)
        a_w = self.sigmoid(self.conv_w(y_w)) # (B, C, 1, 1, W)

        # 3. Artifact Awareness via Attention Re-weighting
        # The gating weights explicitly suppress artifact-corrupted channels 
        # and highlight structurally important regions.
        out = identity * a_d * a_h * a_w

        return out
