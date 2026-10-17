import torch
import torch.nn as nn
import torch.fft

class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (FFL) for Image Reconstruction and Synthesis
    Based on Jiang et al., ICCV 2021.
    
    Operates on 3D patches of shape (B, C, D, H, W).
    Computes 3D FFT explicitly over the (D, H, W) spatial dimensions.
    """
    def __init__(self, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # pred, target: (B, C, D, H, W)
        
        # Apply 3D FFT over the spatial dimensions (D, H, W)
        # norm="ortho" provides unitary normalization
        F_pred = torch.fft.fftn(pred, dim=(-3, -2, -1), norm="ortho")
        F_target = torch.fft.fftn(target, dim=(-3, -2, -1), norm="ortho")
        
        # Complex difference
        F_diff = F_pred - F_target
        
        # Absolute difference squared: |F_pred - F_gt|^2
        diff_sq = torch.abs(F_diff) ** 2
        
        # Dynamic spectrum weight matrix W = |F_pred - F_gt|^alpha
        with torch.no_grad():
            W = torch.abs(F_diff) ** self.alpha
            
            # Normalize W into [0,1] per sample/channel
            W_min = W.amin(dim=(-3, -2, -1), keepdim=True)
            W_max = W.amax(dim=(-3, -2, -1), keepdim=True)
            W = (W - W_min) / (W_max - W_min + 1e-8)
            
        # The weight matrix W is automatically detached because of torch.no_grad()
        loss = torch.mean(W * diff_sq)
        
        return loss
