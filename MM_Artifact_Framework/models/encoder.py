import torch
import torch.nn as nn

class Encoder3D(nn.Module):
    """
    Standard 3D Encoder for extracting features from the HM2 MRI Patch.
    
    Gradually increases the channel dimension while optionally downsampling
    the spatial dimensions.
    """
    def __init__(self, in_channels: int = 1, base_filters: int = 32):
        """
        Args:
            in_channels (int): Number of input channels (e.g., 1 for MRI).
            base_filters (int): Number of filters in the first convolutional layer.
        """
        super(Encoder3D, self).__init__()
        
        # Layer 1: Extract basic features, maintain spatial dimensions
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 2: Downsample spatially by 2 in H and W, maintain D
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 3: Further downsample spatially by 2 in H and W, maintain D
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input patch of shape (B, 1, D, H, W)
            
        Returns:
            tuple: (F_high, F_mid, F_bottleneck)
                   F_high: (B, 32, D, H, W)
                   F_mid: (B, 64, D, H/2, W/2)
                   F_bottleneck: (B, 128, D, H/4, W/4)
        """
        f_high = self.conv1(x)
        f_mid = self.conv2(f_high)
        f_bottleneck = self.conv3(f_mid)
        return f_high, f_mid, f_bottleneck


class LightweightEncoder3D(nn.Module):
    """
    Lightweight 3D Encoder for extracting features from the Predicted Artifact Map.
    
    Since the artifact map contains less structural complexity than the raw MRI,
    this encoder uses fewer parameters to prevent overfitting on the artifact map.
    """
    def __init__(self, in_channels: int = 1, base_filters: int = 16):
        """
        Args:
            in_channels (int): Number of input channels.
            base_filters (int): Number of base filters, usually smaller than main encoder.
        """
        super(LightweightEncoder3D, self).__init__()
        
        # We aim to output the same channel dimensions as the main encoder 
        # at the final layer, so we scale up accordingly or let fusion handle it.
        # Here we map directly to the same target dimension to simplify fusion.
        target_filters = base_filters * 8  # e.g. 16 * 8 = 128 (same as 32 * 4)
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_filters * 2, target_filters, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(target_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Artifact map of shape (B, 1, D, H, W)
            
        Returns:
            torch.Tensor: Encoded features of shape (B, target_filters, D, H/4, W/4)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
