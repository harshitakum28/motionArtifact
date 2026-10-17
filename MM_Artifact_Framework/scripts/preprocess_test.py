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

def process_test_subjects(dataset_path, output_dir, subjects_list, debug_mode=False):
    """
    Reads NIfTI files, corrects orientation, normalizes using SHARED SUBJECT/VOLUME-LEVEL stats 
    (from Clean), extracts patches, and saves specifically for testing.
    """
    for i, subject_id in enumerate(subjects_list):
        subj_anat_dir = os.path.join(dataset_path, subject_id, "anat")
        
        files_map = {
            'clean': f"{subject_id}_acq-standard_T1w.nii.gz",
            'hm1': f"{subject_id}_acq-headmotion1_T1w.nii.gz",
            'hm2': f"{subject_id}_acq-headmotion2_T1w.nii.gz"
        }
        
        # Save directly in test_dataset/<subject_id>
        out_subj_dir = os.path.join(output_dir, subject_id)
        
        os.makedirs(os.path.join(out_subj_dir, "clean"), exist_ok=True)
        os.makedirs(os.path.join(out_subj_dir, "hm1"), exist_ok=True)
        os.makedirs(os.path.join(out_subj_dir, "hm2"), exist_ok=True)
        
        print(f"\n======================================")
        print(f"Processing Test Subject {subject_id}...")
        print(f"======================================")
        
        volumes = {}
        missing_files = False
        for key, nii_name in files_map.items():
            nii_path = os.path.join(subj_anat_dir, nii_name)
            if not os.path.exists(nii_path):
                print(f"File not found: {nii_path}. Make sure source dataset path is correct.")
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
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Test MRI data (16 subjects)")
    parser.add_argument("--src_dir", type=str, default=r"C:\Users\bitpa\OneDrive\Desktop\dataset download\ds004173-download", help="Source directory containing raw NIfTI files")
    parser.add_argument("--dest_dir", type=str, default=r"C:\Users\bitpa\OneDrive\Desktop\cordinate attention module\MRI_Artifact_Framework\test_dataset", help="Output directory")
    args = parser.parse_args()
    
    src_dir = args.src_dir
    dest_dir = args.dest_dir
    
    # Automatically detect all subject folders starting with 'sub-' in the source directory
    subjects = []
    if os.path.exists(src_dir):
        for item in os.listdir(src_dir):
            if item.startswith("sub-") and os.path.isdir(os.path.join(src_dir, item)):
                subjects.append(item)
    
    if not subjects:
        print(f"No subjects found in {src_dir}. Falling back to default list.")
        subjects = [f"sub-{i:03d}" for i in range(1, 17)]
    else:
        subjects.sort()
    
    print(f"Found {len(subjects)} test subjects in configuration. Processing...")
    print(f"Source Directory: {src_dir}")
    print(f"Output Directory: {dest_dir}")
    print(f"Subjects to process: {subjects}")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    print("\nStarting test data preprocessing with Shared Volume-Level Normalization...")
    process_test_subjects(src_dir, dest_dir, subjects, debug_mode=False)
    print("Test data preprocessing complete!")
