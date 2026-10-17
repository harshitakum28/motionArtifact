import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Generates a 1D Gaussian tensor."""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window_3d(window_size: int, channel: int) -> torch.Tensor:
    """Creates a 3D Gaussian window."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.view(1, -1)).view(window_size, window_size, window_size)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

def ssim_3d(img1: torch.Tensor, img2: torch.Tensor, window_size: int, window: torch.Tensor, size_average: bool = True) -> torch.Tensor:
    """Computes the 3D SSIM between two images."""
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=img2.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    """
    Stage 2 Loss: Computes the Structural Similarity Index Loss 
    between the reconstructed MRI patch and the Clean MRI patch.
    
    Formula:
    L_SSIM = 1 - SSIM(I, \hat{I})
    """
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3d(window_size, self.channel)

    def forward(self, pred_clean: torch.Tensor, target_clean: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pred_clean (torch.Tensor): Reconstructed MRI Patch \hat{I}.
            target_clean (torch.Tensor): Ground Truth Clean MRI Patch I.
            
        Returns:
            torch.Tensor: Scalar SSIM loss.
        """
        (_, channel, _, _, _) = pred_clean.size()

        if channel == self.channel and self.window.data.type() == pred_clean.data.type():
            window = self.window
        else:
            window = create_window_3d(self.window_size, channel).to(pred_clean.device).type_as(pred_clean)
            self.window = window
            self.channel = channel

        return 1.0 - ssim_3d(pred_clean, target_clean, self.window_size, window, self.size_average)
