import torch
import torch.nn as nn

class ResidualBlock3D(nn.Module):
    """
    Standard 3D Residual Block for refinement.
    
    This block preserves the spatial dimensions and number of channels.
    It helps in refining high-frequency anatomical details in the MRI patches.
    """
    def __init__(self, channels: int):
        """
        Args:
            channels (int): Number of input and output channels.
        """
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)
            
        Returns:
            torch.Tensor: Refined tensor of shape (B, C, D, H, W)
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the skip connection before the final activation
        out += identity
        out = self.relu(out)
        
        return out

class ResidualGroup3D(nn.Module):
    """
    A sequence of Residual Blocks.
    """
    def __init__(self, channels: int, num_blocks: int = 4):
        """
        Args:
            channels (int): Number of channels.
            num_blocks (int): Number of residual blocks in the group.
        """
        super(ResidualGroup3D, self).__init__()
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock3D(channels))
            
        self.group = nn.Sequential(*blocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.group(x)
