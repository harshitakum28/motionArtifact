import torch
import torch.nn as nn

class FeatureFusion3D(nn.Module):
    """
    Fuses features from the main encoder and lightweight encoder.
    
    Instead of passing directly concatenated features to the rest of the network,
    this module concatenates them along the channel dimension and immediately 
    applies a 1x1x1 Convolution to intelligently blend the HM2 structural 
    features with the Artifact semantic features.
    """
    def __init__(self, channels_hm2: int, channels_artifact: int, out_channels: int):
        """
        Args:
            channels_hm2 (int): Number of channels from the HM2 encoder.
            channels_artifact (int): Number of channels from the Artifact map encoder.
            out_channels (int): Desired output channel dimension after fusion.
        """
        super(FeatureFusion3D, self).__init__()
        
        in_channels = channels_hm2 + channels_artifact
        
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, feature_hm2: torch.Tensor, feature_artifact: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature fusion.
        
        Args:
            feature_hm2 (torch.Tensor): Feature tensor from HM2 encoder.
            feature_artifact (torch.Tensor): Feature tensor from Artifact map encoder.
            
        Returns:
            torch.Tensor: Fused features of shape (B, out_channels, D, H, W)
        """
        # Concatenate along the channel dimension (dim=1)
        concatenated = torch.cat([feature_hm2, feature_artifact], dim=1)
        
        # Fuse using 1x1x1 convolution
        fused = self.fusion(concatenated)
        
        return fused
