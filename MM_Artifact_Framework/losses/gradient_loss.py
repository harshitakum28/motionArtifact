import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientLoss3D(nn.Module):
    """
    3D Spatial Gradient Loss.
    Encourages the spatial derivatives (edges) of the reconstructed image 
    to match the clean image across Depth, Height, and Width.
    """
    def __init__(self):
        super(GradientLoss3D, self).__init__()

    def forward(self, pred, target):
        """
        pred, target: (B, C, D, H, W)
        """
        # Calculate gradients for predictions
        pred_grad_d = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
        pred_grad_h = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        pred_grad_w = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
        
        # Calculate gradients for targets
        target_grad_d = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])
        target_grad_h = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
        target_grad_w = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
        
        # Compute L1 difference between gradients
        loss_d = F.l1_loss(pred_grad_d, target_grad_d)
        loss_h = F.l1_loss(pred_grad_h, target_grad_h)
        loss_w = F.l1_loss(pred_grad_w, target_grad_w)
        
        return loss_d + loss_h + loss_w
