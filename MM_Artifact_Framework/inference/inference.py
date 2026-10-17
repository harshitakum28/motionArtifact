import torch
import numpy as np
from typing import List

class MRIArtifactFrameworkInference:
    """
    End-to-End Inference pipeline for MRI Motion Artifact Removal Framework.
    
    This class orchestrates both Stage 1 (Artifact Prediction) and 
    Stage 2 (Artifact-Guided Reconstruction) to process a full volume.
    
    Critically, it ONLY accepts HM1 and HM2 volumes. It never uses the Clean MRI.
    """
    def __init__(self, stage1_model: torch.nn.Module, stage2_model: torch.nn.Module, device: torch.device):
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.device = device
        
        self.stage1.eval()
        self.stage2.eval()

    @torch.no_grad()
    def infer_patch(self, hm1_patch: torch.Tensor, hm2_patch: torch.Tensor) -> torch.Tensor:
        """
        Runs inference on a single corresponding pair of HM1 and HM2 patch tensors.
        Inputs are expected to be (1, 1, D, H, W) and normalized.
        """
        pred_artifact = self.infer_stage1_patch(hm1_patch, hm2_patch)
        pred_clean = self.infer_stage2_patch(hm2_patch, pred_artifact)
        return pred_clean
        
    @torch.no_grad()
    def infer_stage1_patch(self, hm1_patch: torch.Tensor, hm2_patch: torch.Tensor) -> torch.Tensor:
        """Stage 1: Predict artifact map"""
        return self.stage1(hm1_patch, hm2_patch)
        
    @torch.no_grad()
    def infer_stage2_patch(self, hm2_patch: torch.Tensor, pred_artifact: torch.Tensor) -> torch.Tensor:
        """Stage 2: Reconstruct Clean MRI"""
        # Note: Depending on Stage2 architecture, it might return a tuple if return_correction=True is set.
        # But in default forward pass, it returns just the clean tensor.
        return self.stage2(hm2_patch, pred_artifact)
        
    def infer_volume(self, hm1_volume: torch.Tensor, hm2_volume: torch.Tensor, mode: str = "sliding_window") -> torch.Tensor:
        """
        Runs inference on the 3D volume.
        Inputs: (1, 1, D, H, W) tensors.
        mode: "sliding_window" (overlap-add with Hann weighting).
        Returns: Reconstructed volume tensor of shape (1, 1, D, H, W).
        """
        _, _, d, h, w = hm1_volume.shape
        reconstructed_vol = torch.zeros_like(hm1_volume)
        if mode == "full_volume":
            print("Running full volume inference directly (WARNING: May cause CUDA OutOfMemory)...")
            pred_clean = self.infer_patch(hm1_volume, hm2_volume)
            return pred_clean
            
        elif mode == "training_style":
            print("Running training-style patch inference (exactly 16 non-overlapping patches)...")
            if d != 256:
                raise ValueError(f"training_style expects volume depth of 256, got {d}")
                
            for i in range(16):
                start_z = i * 16
                end_z = start_z + 16
                
                hm1_patch = hm1_volume[:, :, start_z:end_z, :, :]
                hm2_patch = hm2_volume[:, :, start_z:end_z, :, :]
                
                pred_clean_patch = self.infer_patch(hm1_patch, hm2_patch)
                
                # Direct placement (no overlap)
                reconstructed_vol[:, :, start_z:end_z, :, :] = pred_clean_patch
                

        elif mode == "sliding_window":
            # Sliding window with Hann weighting (overlap-add)
            stride = 8
            window_size = 16
            
            # Create a 1D Hann window along the depth dimension
            hann = torch.hann_window(window_size).view(1, 1, window_size, 1, 1).to(self.device)
            weight_vol = torch.zeros_like(hm1_volume)
            
            # Determine start indices for all windows
            starts = list(range(0, d - window_size + 1, stride))
            
            # Ensure the very end of the volume is covered if it wasn't perfectly aligned
            if starts[-1] + window_size < d:
                starts.append(d - window_size)
                
            num_steps = len(starts)
            print(f"Running sliding window patch inference ({num_steps} steps) with Hann weighting...")
            
            for start_z in starts:
                end_z = start_z + window_size
                
                hm1_patch = hm1_volume[:, :, start_z:end_z, :, :]
                hm2_patch = hm2_volume[:, :, start_z:end_z, :, :]
                
                pred_clean_patch = self.infer_patch(hm1_patch, hm2_patch)
                
                # Overlap-add
                reconstructed_vol[:, :, start_z:end_z, :, :] += pred_clean_patch * hann
                weight_vol[:, :, start_z:end_z, :, :] += hann
                
            # Normalize by weights (add small epsilon to avoid div by zero)
            reconstructed_vol = reconstructed_vol / (weight_vol + 1e-8)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        return reconstructed_vol
