import torch
import torch.nn as nn

class Decoder3D(nn.Module):
    """
    Standard 3D Decoder for reconstructing the Clean MRI Patch.
    
    Gradually decreases the channel dimension while upsampling
    the spatial dimensions (H and W) back to the original resolution
    using learnable ConvTranspose3d operations.
    """
    def __init__(self, in_channels: int, out_channels: int = 1):
        """
        Args:
            in_channels (int): Number of input channels (from residual blocks).
            out_channels (int): Number of output channels (e.g., 1 for Clean MRI).
        """
        super(Decoder3D, self).__init__()
        
        # Layer 1: Upsample using learnable ConvTranspose3d
        self.up1 = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Skip Fusion 1: Concatenates F_mid (64 channels) with up1 (64 channels) -> 128 channels. Fuses down to 64 channels.
        self.skip_fusion1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels // 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 2: Further upsample using learnable ConvTranspose3d
        self.up2 = nn.ConvTranspose3d(in_channels // 2, in_channels // 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Skip Fusion 2: Concatenates F_high (32 channels) with up2 (32 channels) -> 64 channels. Fuses down to 32 channels.
        self.skip_fusion2 = nn.Sequential(
            nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels // 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 3: Final mapping to desired output channels, maintain spatial dims
        self.conv_final = nn.Sequential(
            nn.Conv3d(in_channels // 4, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, f_high: torch.Tensor, f_mid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections.
        
        Args:
            x (torch.Tensor): Refined features of shape (B, in_channels, D, H/4, W/4)
            f_high (torch.Tensor): Skip features from encoder of shape (B, 32, D, H, W)
            f_mid (torch.Tensor): Skip features from encoder of shape (B, 64, D, H/2, W/2)
            
        Returns:
            torch.Tensor: Reconstructed MRI patch of shape (B, 1, D, H, W)
        """
        # Upsample 1
        x1 = self.up1(x)
        # Pad if necessary to match f_mid (usually exact if dimensions are divisible)
        diffY = f_mid.size()[3] - x1.size()[3]
        diffX = f_mid.size()[4] - x1.size()[4]
        if diffX != 0 or diffY != 0:
            import torch.nn.functional as F
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        x1_fused = self.skip_fusion1(torch.cat([x1, f_mid], dim=1))
        
        # Upsample 2
        x2 = self.up2(x1_fused)
        diffY = f_high.size()[3] - x2.size()[3]
        diffX = f_high.size()[4] - x2.size()[4]
        if diffX != 0 or diffY != 0:
            import torch.nn.functional as F
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        x2_fused = self.skip_fusion2(torch.cat([x2, f_high], dim=1))
        
        out = self.conv_final(x2_fused)
        return out
