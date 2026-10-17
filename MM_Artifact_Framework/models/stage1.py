import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with strided Conv3D then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Only downsample spatial dimensions H and W to preserve D
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample spatial dims H and W
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if necessary (though patches should be exactly divisible)
        # input is CHW
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        import torch.nn.functional as F
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Stage1ArtifactPredictor(nn.Module):
    """
    Stage 1: Predicts the Artifact Map from HM1 and HM2 patches.
    
    Architecture: 3D U-Net variant tailored for spatial patch preservation.
    Input: Concatenation of HM1 and HM2 (Channels = 2)
    Output: Predicted Artifact Map (Channels = 1)
    """
    def __init__(self, in_channels=2, out_channels=1, base_filters=16):
        super(Stage1ArtifactPredictor, self).__init__()
        
        self.inc = DoubleConv3D(in_channels, base_filters)
        self.down1 = Down3D(base_filters, base_filters * 2)
        self.down2 = Down3D(base_filters * 2, base_filters * 4)
        
        self.up1 = Up3D(base_filters * 4, base_filters * 2)
        self.up2 = Up3D(base_filters * 2, base_filters)
        
        self.outc = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, hm1: torch.Tensor, hm2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hm1 (torch.Tensor): HM1 patch (B, 1, D, H, W)
            hm2 (torch.Tensor): HM2 patch (B, 1, D, H, W)
            
        Returns:
            torch.Tensor: Predicted Artifact Map (B, 1, D, H, W)
        """
        # Concatenate inputs along channel dimension
        x = torch.cat([hm1, hm2], dim=1) # (B, 2, D, H, W)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        
        logits = self.outc(x)
        
        # Typically artifact maps represent residuals (which can be positive or negative)
        # We don't constrain the output with a sigmoid/tanh unless data is normalized accordingly.
        return logits
