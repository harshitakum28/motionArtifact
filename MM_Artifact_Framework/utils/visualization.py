import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a tensor to a numpy array for visualization."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def plot_mri_comparison(hm1, hm2, clean, pred_clean, save_path, slice_idx=None):
    """
    Plots a side-by-side comparison of MRI slices.
    Expects 3D volumes (D, H, W) or (1, D, H, W).
    """
    hm1_np = tensor_to_numpy(hm1).squeeze()
    hm2_np = tensor_to_numpy(hm2).squeeze()
    clean_np = tensor_to_numpy(clean).squeeze()
    pred_np = tensor_to_numpy(pred_clean).squeeze()
    
    # Pick middle slice along the third dimension (depth) if not provided
    if slice_idx is None:
        slice_idx = hm1_np.shape[2] // 2
        
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(hm1_np[:, :, slice_idx], cmap='gray')
    axes[0].set_title("Artifact (HM1)")
    axes[0].axis('off')
    
    axes[1].imshow(hm2_np[:, :, slice_idx], cmap='gray')
    axes[1].set_title("Artifact (HM2)")
    axes[1].axis('off')
    
    axes[2].imshow(pred_np[:, :, slice_idx], cmap='gray')
    axes[2].set_title("Reconstructed")
    axes[2].axis('off')
    
    axes[3].imshow(clean_np[:, :, slice_idx], cmap='gray')
    axes[3].set_title("Ground Truth (Clean)")
    axes[3].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")

def plot_axial_slice(volume, save_path, slice_idx=None, title="Axial Slice (X-Y Plane)"):
    """
    Plots an axial slice (X-Y plane) of a 3D MRI volume.
    Assumes the volume has shape (Z, Y, X).
    """
    vol_np = tensor_to_numpy(volume).squeeze()
    
    # Pick middle slice along the Z dimension if not provided
    if slice_idx is None:
        slice_idx = vol_np.shape[0] // 2
        
    plt.figure(figsize=(6, 6))
    
    # The X-Y plane corresponds to a fixed Z index
    plt.imshow(vol_np[slice_idx, :, :], cmap='gray')
    plt.title(f"{title} - Z: {slice_idx}")
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Axial slice visualization saved to {save_path}")

def plot_mri_clamped_comparison(hm2, pred_raw, pred_clamped, clean, save_path, slice_idx=None):
    """
    Plots HM2, Raw Reconstruction, Clamped Reconstruction, and Ground Truth.
    Uses identical display limits (vmin=0.0, vmax=1.0) to prevent matplotlib auto-scaling.
    Expects 3D volumes (D, H, W) or (1, D, H, W).
    """
    hm2_np = tensor_to_numpy(hm2).squeeze()
    pred_raw_np = tensor_to_numpy(pred_raw).squeeze()
    pred_clamped_np = tensor_to_numpy(pred_clamped).squeeze()
    clean_np = tensor_to_numpy(clean).squeeze()
    
    # Pick middle slice along the third dimension (depth) if not provided
    if slice_idx is None:
        slice_idx = hm2_np.shape[2] // 2
        
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(hm2_np[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
    axes[0].set_title("Artifact (HM2)")
    axes[0].axis('off')
    
    axes[1].imshow(pred_raw_np[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
    axes[1].set_title("Raw Reconstruction")
    axes[1].axis('off')
    
    axes[2].imshow(pred_clamped_np[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
    axes[2].set_title("Clamped Reconstruction")
    axes[2].axis('off')
    
    axes[3].imshow(clean_np[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
    axes[3].set_title("Ground Truth (Clean)")
    axes[3].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")

def plot_artifact_residual_comparison(hm2, pred_res, gt_res, pred_clean, clean, save_path, slice_idx=None):
    """
    Plots HM2, Predicted Artifact Residual, Ground-Truth Artifact Residual, Final Reconstruction, Clean Ground Truth.
    HM2, pred_clean, and clean use [0.0, 1.0].
    Residuals use symmetric ranges [-0.5, 0.5] or similar based on data.
    """
    hm2_np = tensor_to_numpy(hm2).squeeze()
    pred_res_np = tensor_to_numpy(pred_res).squeeze()
    gt_res_np = tensor_to_numpy(gt_res).squeeze()
    pred_clean_np = tensor_to_numpy(pred_clean).squeeze()
    clean_np = tensor_to_numpy(clean).squeeze()
    
    if slice_idx is None:
        slice_idx = hm2_np.shape[2] // 2
        
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(hm2_np[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
    axes[0].set_title("Artifact (HM2)")
    axes[0].axis('off')
    
    # Symmetric range for residuals
    res_max = max(abs(gt_res_np[:, :, slice_idx].min()), abs(gt_res_np[:, :, slice_idx].max()))
    if res_max == 0: res_max = 0.5
    
    axes[1].imshow(pred_res_np[:, :, slice_idx], cmap='bwr', vmin=-res_max, vmax=res_max)
    axes[1].set_title("Predicted Residual")
    axes[1].axis('off')
    
    axes[2].imshow(gt_res_np[:, :, slice_idx], cmap='bwr', vmin=-res_max, vmax=res_max)
    axes[2].set_title("GT Residual")
    axes[2].axis('off')
    
    axes[3].imshow(pred_clean_np[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
    axes[3].set_title("Reconstructed")
    axes[3].axis('off')
    
    axes[4].imshow(clean_np[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
    axes[4].set_title("Ground Truth (Clean)")
    axes[4].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")

def plot_strong_residual_comparison(hm2, gt_diff, model_corr, pred_clean, clean, save_path, slice_idx=None):
    """
    Plots HM2, Absolute GT Diff, Absolute Model Correction, Reconstructed, Clean.
    """
    hm2_np = tensor_to_numpy(hm2).squeeze()
    gt_diff_np = np.abs(tensor_to_numpy(gt_diff).squeeze())
    model_corr_np = np.abs(tensor_to_numpy(model_corr).squeeze())
    pred_clean_np = tensor_to_numpy(pred_clean).squeeze()
    clean_np = tensor_to_numpy(clean).squeeze()
    
    if slice_idx is None:
        slice_idx = hm2_np.shape[2] // 2
        
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(hm2_np[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
    axes[0].set_title("Artifact (HM2)")
    axes[0].axis('off')
    
    # Use jet or hot for absolute differences to make them stand out
    diff_vmax = max(gt_diff_np[:, :, slice_idx].max(), model_corr_np[:, :, slice_idx].max())
    if diff_vmax == 0: diff_vmax = 0.1
    
    axes[1].imshow(gt_diff_np[:, :, slice_idx], cmap='hot', vmin=0.0, vmax=diff_vmax)
    axes[1].set_title("Absolute GT Diff")
    axes[1].axis('off')
    
    axes[2].imshow(model_corr_np[:, :, slice_idx], cmap='hot', vmin=0.0, vmax=diff_vmax)
    axes[2].set_title("Absolute Model Corr")
    axes[2].axis('off')
    
    axes[3].imshow(pred_clean_np[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
    axes[3].set_title("Reconstructed")
    axes[3].axis('off')
    
    axes[4].imshow(clean_np[:, :, slice_idx], cmap='gray', vmin=0.0, vmax=1.0)
    axes[4].set_title("Ground Truth (Clean)")
    axes[4].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")
