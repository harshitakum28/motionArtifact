import torch
import torch.nn as nn

class ArtifactAwarePreservationLoss(nn.Module):
    """
    Artifact-Aware Preservation Loss.
    Strongly penalizes the network for applying any non-zero correction 
    in regions where the HM2 image is already identical (or very close) to the Clean image.
    """
    def __init__(self, k=50.0):
        super(ArtifactAwarePreservationLoss, self).__init__()
        self.k = k

    def forward(self, raw_correction, hm2, clean):
        """
        raw_correction: Predicted correction (B, C, D, H, W)
        hm2: Input degraded image (B, C, D, H, W)
        clean: Target clean image (B, C, D, H, W)
        """
        # Calculate true artifact magnitude
        # Detach to ensure we don't accidentally push gradients through the target calculation
        with torch.no_grad():
            true_artifact_mag = torch.abs(hm2 - clean)
            
            # The weight is 1.0 when artifact is 0.0 (perfectly clean tissue)
            # The weight drops exponentially to 0.0 as the artifact grows
            low_artifact_weight = torch.exp(-self.k * true_artifact_mag)
            
        # Penalize correction magnitude, scaled by how clean the region already is
        loss = torch.mean(low_artifact_weight * torch.abs(raw_correction))
        
        return loss
