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

def process_subjects(dataset_path, output_dir, subjects_list, debug_mode=False):
    """
    Reads NIfTI files, corrects orientation, normalizes using SHARED SUBJECT/VOLUME-LEVEL stats 
    (from Clean), extracts patches, and saves.
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
            print(f"[{key}] Original shape: {original_shape}")
            
            # Strict orientation correction (Dataset reality check)
            if original_shape == (192, 256, 256):
                print(f"[{key}] Shape is already (192, 256, 256). No transpose or padding needed.")
            else:
                raise ValueError(f"Unexpected NIfTI shape {original_shape} for {key}. Expected (192, 256, 256).")
            
            # Final validation/resize
            if data.shape != (192, 256, 256):
                data = resize_volume(data, target_shape=(192, 256, 256)).astype(np.float32)
                print(f"[{key}] Resized shape: {data.shape}")
            else:
                data = data.astype(np.float32)
                
            volumes[key] = data
            
        if missing_files:
            continue
            
        # Verify shape consistency between the 3 modalities
        v_shapes = [vol.shape for vol in volumes.values()]
        if not all(s == v_shapes[0] for s in v_shapes):
            raise ValueError(f"Shape mismatch between modalities after orientation correction: {v_shapes}")
            
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
            
        # 2. Apply SAME values to all three volumes and extract patches
        for key, vol in volumes.items():
            # Clip and Normalize
            vol_clipped = np.clip(vol, low, high)
            vol_norm = (vol_clipped - low) / (high - low + 1e-8)
            
            if debug_mode:
                print(f"[{key}] Normalized volume shape: {vol_norm.shape}")
            
            np.save(os.path.join(out_subj_dir, f"{key}.npy"), vol_norm.astype(np.float32))
            
            # Extract patches AFTER volume-level normalization
            patches = extract_patches(vol_norm, num_patches=16)
            
            if debug_mode:
                print(f"[{key}] Generated {len(patches)} patches of shape {patches[0].shape}")
            
            # Save patches
            for p_idx, patch in enumerate(patches):
                patch_name = f"{key}_patch_{p_idx:02d}.npy"
                np.save(os.path.join(out_subj_dir, key, patch_name), patch.astype(np.float32))
                
        print(f"Successfully processed and normalized patches for {subject_id}")
        
        if debug_mode:
            print("\nDebug mode: saving visual slices...")
            slice_idx = 256 // 2
            
            clean_slice = volumes['clean'][:, :, slice_idx]
            hm1_slice = volumes['hm1'][:, :, slice_idx]
            hm2_slice = volumes['hm2'][:, :, slice_idx]
            
            def norm_vis(x):
                xc = np.clip(x, low, high)
                return (xc - low) / (high - low + 1e-8)
                
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(norm_vis(clean_slice), cmap='gray', vmin=0, vmax=1)
            axes[0].set_title("Clean (Middle Slice)")
            axes[0].axis('off')
            
            axes[1].imshow(norm_vis(hm1_slice), cmap='gray', vmin=0, vmax=1)
            axes[1].set_title("HM1 (Middle Slice)")
            axes[1].axis('off')
            
            axes[2].imshow(norm_vis(hm2_slice), cmap='gray', vmin=0, vmax=1)
            axes[2].set_title("HM2 (Middle Slice)")
            axes[2].axis('off')
            
            plt.tight_layout()
            vis_path = os.path.join(output_dir, "debug_visuals.png")
            plt.savefig(vis_path)
            plt.close()
            print(f"Saved visual diagnostics to {vis_path}")
            print("Debug single subject processing complete. Exiting early.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MRI data with Shared Volume-Level Normalization")
    parser.add_argument("--debug_subject", action="store_true", help="Process only the first subject and save diagnostics")
    args = parser.parse_args()
    
    src_dir = r"C:\Users\bitpa\OneDrive\Desktop\dataset download\ds004173-download"
    dest_dir = r"C:\Users\bitpa\OneDrive\Desktop\kaggle dataset"
    
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
    
    print("Starting data preprocessing with Shared Volume-Level Normalization...")
    process_subjects(src_dir, dest_dir, subjects, debug_mode=args.debug_subject)
    print("Data preprocessing complete!")
