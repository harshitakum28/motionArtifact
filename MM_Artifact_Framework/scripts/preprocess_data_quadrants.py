import os
import json
import nibabel as nib
import numpy as np
import argparse
import matplotlib.pyplot as plt

def resize_volume(volume, target_shape=(192, 256, 256)):
    """Center crops or pads a volume to the target shape."""
    curr_shape = volume.shape
    pad_dims = []
    for c, t in zip(curr_shape, target_shape):
        if c < t:
            pad_before = (t - c) // 2
            pad_after = t - c - pad_before
            pad_dims.append((pad_before, pad_after))
        else:
            pad_dims.append((0, 0))
            
    if any(sum(p) > 0 for p in pad_dims):
        volume = np.pad(volume, pad_dims, mode='constant', constant_values=0)
        
    curr_shape = volume.shape
    crop_slices = []
    for c, t in zip(curr_shape, target_shape):
        if c > t:
            start = (c - t) // 2
            crop_slices.append(slice(start, start + t))
        else:
            crop_slices.append(slice(None))
            
    volume = volume[tuple(crop_slices)]
    return volume

def extract_patches(volume, num_patches=16):
    """Divides a 3D volume into non-overlapping patches along depth."""
    h, w, d = volume.shape
    patch_depth = d // num_patches
    patches = []
    for i in range(num_patches):
        start_idx = i * patch_depth
        end_idx = (i + 1) * patch_depth
        patches.append(volume[:, :, start_idx:end_idx])
    return patches

def extract_quadrant_patches(patch_3d):
    """
    Splits a 3D patch (e.g. 192, 256, 16) into 4 spatial quadrants.
    Returns a dictionary of the 4 quadrants (TL, TR, BL, BR).
    """
    h, w, d = patch_3d.shape
    h2, w2 = h // 2, w // 2
    
    # Top-Left, Top-Right, Bottom-Left, Bottom-Right
    tl = patch_3d[:h2, :w2, :]
    tr = patch_3d[:h2, w2:, :]
    bl = patch_3d[h2:, :w2, :]
    br = patch_3d[h2:, w2:, :]
    
    return {'TL': tl, 'TR': tr, 'BL': bl, 'BR': br}

def process_subjects(dataset_path, output_dir, subjects_list, debug_mode=False):
    """
    Reads NIfTI files, normalizes, extracts Z-patches, then splits each patch into 4 quadrants (X-Y).
    """
    for i, subject_id in enumerate(subjects_list):
        subj_anat_dir = os.path.join(dataset_path, subject_id, "anat")
        
        files_map = {
            'clean': f"{subject_id}_acq-standard_T1w.nii.gz",
            'hm1': f"{subject_id}_acq-headmotion1_T1w.nii.gz",
            'hm2': f"{subject_id}_acq-headmotion2_T1w.nii.gz"
        }
        
        train_count = int(0.8 * len(subjects_list))
        split = "train" if i < train_count else "val"
        out_subj_dir = os.path.join(output_dir, "data", split, f"subject_{i+1:02d}")
        
        os.makedirs(os.path.join(out_subj_dir, "clean"), exist_ok=True)
        os.makedirs(os.path.join(out_subj_dir, "hm1"), exist_ok=True)
        os.makedirs(os.path.join(out_subj_dir, "hm2"), exist_ok=True)
        
        print(f"\n======================================")
        print(f"Processing {subject_id} into {split} split...")
        print(f"======================================")
        
        volumes = {}
        missing_files = False
        for key, nii_name in files_map.items():
            nii_path = os.path.join(subj_anat_dir, nii_name)
            if not os.path.exists(nii_path):
                print(f"File not found: {nii_path}")
                missing_files = True
                break
                
            img = nib.load(nii_path)
            data = img.get_fdata()
            
            # Handle singleton dimension
            if data.ndim == 4:
                data = np.squeeze(data, axis=3)
                
            original_shape = data.shape
            
            if data.shape != (192, 256, 256):
                data = resize_volume(data, target_shape=(192, 256, 256)).astype(np.float32)
            else:
                data = data.astype(np.float32)
                
            volumes[key] = data
            
        if missing_files:
            continue
            
        # 1. Compute robust normalization statistics ONCE per subject using CLEAN
        clean_vol = volumes['clean']
        low = float(np.percentile(clean_vol, 0.5))
        high = float(np.percentile(clean_vol, 99.5))
        
        print(f"Shared Normalization Stats (Clean reference) - Low: {low:.4f}, High: {high:.4f}")
        
        # Save metadata
        meta = {
            "subject_id": subject_id,
            "reference": "clean",
            "low": low,
            "high": high
        }
        with open(os.path.join(out_subj_dir, "norm_meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
            
        # 2. Apply SAME values to all three volumes and extract quadrant patches
        for key, vol in volumes.items():
            # Clip and Normalize
            vol_clipped = np.clip(vol, low, high)
            vol_norm = (vol_clipped - low) / (high - low + 1e-8)
            
            # Save full normalized volume
            np.save(os.path.join(out_subj_dir, f"{key}.npy"), vol_norm.astype(np.float32))
            
            # Extract Z-axis patches
            z_patches = extract_patches(vol_norm, num_patches=16)
            
            # For each Z patch, extract 4 spatial quadrants
            total_saved = 0
            for p_idx, z_patch in enumerate(z_patches):
                quadrants = extract_quadrant_patches(z_patch)
                
                for q_name, q_patch in quadrants.items():
                    patch_name = f"{key}_patch_{p_idx:02d}_{q_name}.npy"
                    np.save(os.path.join(out_subj_dir, key, patch_name), q_patch.astype(np.float32))
                    total_saved += 1
                    
            if debug_mode:
                print(f"[{key}] Generated {total_saved} quadrant patches of shape {q_patch.shape}")
                
        print(f"Successfully processed and normalized quadrant patches for {subject_id}")
        
        if debug_mode:
            print("Debug single subject processing complete. Exiting early.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MRI data with Quadrant Patch Split")
    parser.add_argument("--debug_subject", action="store_true", help="Process only the first subject and save diagnostics")
    args = parser.parse_args()
    
    src_dir = r"C:\Users\bitpa\OneDrive\Desktop\dataset download\ds004173-download"
    # NEW output directory for quadrant dataset
    dest_dir = r"C:\Users\bitpa\OneDrive\Desktop\kaggle dataset_quadrants"
    
    # Automatically discover all subjects in the source directory
    import glob
    subject_dirs = sorted(glob.glob(os.path.join(src_dir, "sub-*")))
    subjects = [os.path.basename(d) for d in subject_dirs]
    
    if not subjects:
        print(f"No subjects found in {src_dir}. Please check the path.")
        exit(1)
        
    print(f"Found {len(subjects)} subjects. Processing...")
    
    # Shuffle predictably to ensure consistent train/val split
    import random
    random.seed(42)
    random.shuffle(subjects)
    
    print(f"Starting data preprocessing... Output will be saved to: {dest_dir}")
    process_subjects(src_dir, dest_dir, subjects, debug_mode=args.debug_subject)
    print("Quadrant Data preprocessing complete!")
