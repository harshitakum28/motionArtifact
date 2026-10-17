import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .transforms import ToTensor

class MRIMotionArtifactDataset(Dataset):
    """
    PyTorch Dataset for MRI Motion Artifact Framework.
    
    Loads Clean MRI, HM1, and HM2 patches for each subject.
    Note: Patches are expected to be ALREADY NORMALIZED using Shared Subject-Level stats.
    """
    def __init__(self, data_dir: str = None, subject_dirs: list = None, transform=None):
        if data_dir is None and subject_dirs is None:
            raise ValueError("Must provide either data_dir or subject_dirs.")
            
        if subject_dirs is not None:
            raw_dirs = subject_dirs
        else:
            raw_dirs = sorted(glob.glob(os.path.join(data_dir, "sub-*"))) + \
                       sorted(glob.glob(os.path.join(data_dir, "subject_*")))
            
        self.valid_patches = []
        
        for subj_dir in raw_dirs:
            clean_dir = os.path.join(subj_dir, 'clean')
            hm1_dir = os.path.join(subj_dir, 'hm1')
            hm2_dir = os.path.join(subj_dir, 'hm2')
            
            if not (os.path.exists(clean_dir) and os.path.exists(hm1_dir) and os.path.exists(hm2_dir)):
                print(f"Warning: Skipping incomplete subject {subj_dir}. Missing clean, hm1, or hm2 folder.")
                continue
                
            clean_patches = sorted(glob.glob(os.path.join(clean_dir, 'clean_patch_*.npy')))
            
            if len(clean_patches) == 0:
                print(f"Warning: Skipping subject {subj_dir} due to no clean patches found.")
                continue
                
            subject_valid_patches = []
            valid_subject = True
            
            for clean_path in clean_patches:
                basename = os.path.basename(clean_path)
                idx_str = basename.replace('clean_patch_', '').replace('.npy', '')
                
                hm1_path = os.path.join(hm1_dir, f'hm1_patch_{idx_str}.npy')
                hm2_path = os.path.join(hm2_dir, f'hm2_patch_{idx_str}.npy')
                
                if os.path.exists(hm1_path) and os.path.exists(hm2_path):
                    subject_valid_patches.append((clean_path, hm1_path, hm2_path))
                else:
                    print(f"Warning: Skipping subject {subj_dir} due to missing hm1 or hm2 patch {idx_str}.")
                    valid_subject = False
                    break
                    
            if valid_subject:
                self.valid_patches.extend(subject_valid_patches)
        
        if len(self.valid_patches) == 0:
            print("Warning: No valid patch triplets found. Check your dataset path and structure.")

        if transform is None:
            self.transform = transforms.Compose([
                ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.valid_patches)

    def __getitem__(self, idx: int):
        clean_path, hm1_path, hm2_path = self.valid_patches[idx]
        
        sample = {
            'clean': np.load(clean_path),
            'hm1': np.load(hm1_path),
            'hm2': np.load(hm2_path)
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

