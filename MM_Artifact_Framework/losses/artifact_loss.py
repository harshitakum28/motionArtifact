import torch
import torch.nn as nn

class ArtifactLoss(nn.Module):
    """
    Stage 1 Loss: Computes the L1 difference between the Predicted Artifact Map
    and the Ground Truth Artifact Map (HM2 - Clean).
    
    Formula:
    L_artifact = 1/N * sum(|ArtifactGT - \hat{A}|)
    """
    def __init__(self):
        super(ArtifactLoss, self).__init__()
        self.criterion = nn.L1Loss()
        
    def forward(self, pred_artifact: torch.Tensor, hm2: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pred_artifact (torch.Tensor): Predicted artifact map \hat{A}.
            hm2 (torch.Tensor): HM2 MRI Patch.
            clean (torch.Tensor): Clean MRI Patch.
            
        Returns:
            torch.Tensor: Scalar L1 loss.
        """
        artifact_gt = hm2 - clean
        loss = self.criterion(pred_artifact, artifact_gt)
        return loss
