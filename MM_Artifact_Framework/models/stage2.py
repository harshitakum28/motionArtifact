import torch
import torch.nn as nn

from .encoder import Encoder3D, LightweightEncoder3D
from .feature_fusion import FeatureFusion3D
from .coord_attention import ArtifactAwareCoordinateAttention
from .residual_blocks import ResidualGroup3D
from .decoder import Decoder3D

class Stage2ArtifactGuidedReconstructor(nn.Module):
    """
    Stage 2: Artifact-Guided MRI Reconstruction.
    
    This network uses the HM2 patch and the Predicted Artifact Map (from Stage 1)
    to reconstruct the Clean MRI patch. The Clean MRI is NEVER used as an input.
    
    Pipeline:
    HM2 -> Encoder -> Feature_HM2
    Artifact Map -> Lightweight Encoder -> Feature_Artifact
    Feature_HM2 + Feature_Artifact -> 1x1 Fusion -> Coordinate Attention -> Residual Blocks -> Decoder -> Clean MRI
    """
    def __init__(self, 
                 base_filters: int = 32, 
                 num_residual_blocks: int = 4):
        super(Stage2ArtifactGuidedReconstructor, self).__init__()
        
        # 1. Encoders
        self.encoder_hm2 = Encoder3D(in_channels=1, base_filters=base_filters)
        
        # Lightweight encoder uses fewer base filters (e.g., base_filters // 2)
        # but outputs the same final target dimension (base_filters * 4) 
        # so they can be easily fused. (Handled internally in LightweightEncoder3D)
        self.encoder_artifact = LightweightEncoder3D(in_channels=1, base_filters=base_filters // 2)
        
        hm2_out_channels = base_filters * 4
        artifact_out_channels = (base_filters // 2) * 8 # = base_filters * 4
        
        # We'll fuse them down to base_filters * 4
        fused_channels = base_filters * 4
        
        # 2. Feature Fusion
        self.fusion = FeatureFusion3D(channels_hm2=hm2_out_channels, 
                                      channels_artifact=artifact_out_channels, 
                                      out_channels=fused_channels)
        
        # 3. Artifact-Aware Coordinate Attention
        self.coord_attention = ArtifactAwareCoordinateAttention(channels=fused_channels, reduction=16)
        
        # 4. Residual Refinement Blocks
        self.residual_group = ResidualGroup3D(channels=fused_channels, num_blocks=num_residual_blocks)
        
        # 5. Decoder
        self.decoder = Decoder3D(in_channels=fused_channels, out_channels=1)

    def forward(self, hm2: torch.Tensor, pred_artifact_map: torch.Tensor, return_correction: bool = False):
        """
        Forward pass.
        
        Args:
            hm2 (torch.Tensor): HM2 MRI Patch (B, 1, D, H, W)
            pred_artifact_map (torch.Tensor): Predicted artifact map from Stage 1 (B, 1, D, H, W)
            return_correction (bool): If True, returns (final_out, raw_correction).
            
        Returns:
            torch.Tensor or tuple: Reconstructed Clean MRI Patch (B, 1, D, H, W) 
                                   or (Clean, Correction)
        """
        # Encode HM2 and get skip features
        f_high, f_mid, f_bottleneck = self.encoder_hm2(hm2)
        
        # Encode artifact map
        feat_artifact = self.encoder_artifact(pred_artifact_map)
        
        # Fuse features at bottleneck
        fused = self.fusion(f_bottleneck, feat_artifact)
        
        # Apply artifact-aware coordinate attention
        attended = self.coord_attention(fused)
        
        # Refine through residual blocks
        refined = self.residual_group(attended)
        
        # Reconstruct clean MRI as HM2 + correction
        raw_correction = self.decoder(refined, f_high, f_mid)
        
        final_out = hm2 + raw_correction
        
        if return_correction:
            return final_out, raw_correction
        return final_out
