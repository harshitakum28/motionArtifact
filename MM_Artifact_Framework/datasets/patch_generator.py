import numpy as np
import torch
from typing import List, Tuple

def extract_patches(volume: np.ndarray, num_patches: int = 16) -> List[np.ndarray]:
    """
    Divides a 3D volume into non-overlapping patches along the depth dimension.
    
    Args:
        volume (np.ndarray): 3D volume of shape (196, 256, 256) (H, W, D).
        num_patches (int): Number of patches to extract.
        
    Returns:
        List[np.ndarray]: List of 16 patches, each of shape (196, 256, 16).
    """
    h, w, d = volume.shape
    assert d % num_patches == 0, f"Depth {d} is not evenly divisible by {num_patches} patches."
    
    patch_depth = d // num_patches
    patches = []
    
    for i in range(num_patches):
        start_idx = i * patch_depth
        end_idx = (i + 1) * patch_depth
        patch = volume[:, :, start_idx:end_idx]
        patches.append(patch)
        
    return patches

def reconstruct_volume(patches: List[np.ndarray]) -> np.ndarray:
    """
    Reconstructs a full 3D volume from a list of patches.
    
    Args:
        patches (List[np.ndarray]): List of 16 patches of shape (196, 256, 16).
        
    Returns:
        np.ndarray: Reconstructed volume of shape (196, 256, 256).
    """
    return np.concatenate(patches, axis=2)
