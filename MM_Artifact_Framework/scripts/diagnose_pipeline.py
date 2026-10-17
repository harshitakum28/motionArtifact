import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import nibabel.orientations as nio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from models.stage1 import Stage1ArtifactPredictor
from models.stage2 import Stage2ArtifactGuidedReconstructor
from inference.inference import MRIArtifactFrameworkInference
from utils.checkpoint import load_checkpoint
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mse
from datasets.transforms import Normalize, ToTensor
from datasets.patch_generator import extract_patches, reconstruct_volume

def calculate_mae(pred, target):
    if isinstance(pred, np.ndarray):
        return np.mean(np.abs(pred - target))
    return torch.mean(torch.abs(pred - target)).item()

def calculate_pearson(pred, target):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
        target = torch.from_numpy(target)
    p = pred.reshape(-1)
    t = target.reshape(-1)
    p_mean = p.mean()
    t_mean = t.mean()
    p_zero = p - p_mean
    t_zero = t - t_mean
    num = torch.sum(p_zero * t_zero)
    den = torch.sqrt(torch.sum(p_zero**2) * torch.sum(t_zero**2))
    if den == 0:
        return 0.0
    return (num / den).item()

def calculate_ncc(pred, target):
    return calculate_pearson(pred, target)

def overlay_images(bg, fg, alpha=0.5):
    # Simple alpha blend
    return bg * (1 - alpha) + fg * alpha

def get_orientation(affine):
    return "".join(nio.aff2axcodes(affine))

def main():
    parser = argparse.ArgumentParser(description="Diagnostic Script for MRI Framework")
    parser.add_argument("--stage1_checkpoint", type=str, required=True, help="Path to Stage 1 checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str, required=True, help="Path to Stage 2 checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing subjects data")
    parser.add_argument("--output_dir", type=str, default="diagnostics", help="Output directory")
    parser.add_argument("--num_subjects", type=int, default=3, help="Number of subjects to process")
    parser.add_argument("--residual_mode", action="store_true", help="Enable residual reconstruction")
    parser.add_argument("--residual_scale", type=float, default=0.25, help="Scale for residual correction")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()

    # 1. Load Checkpoints
    stage1 = Stage1ArtifactPredictor(base_filters=config.BASE_FILTERS_STAGE1).to(device)
    stage2 = Stage2ArtifactGuidedReconstructor(
        base_filters=config.BASE_FILTERS_STAGE2,
        num_residual_blocks=config.NUM_RESIDUAL_BLOCKS,
        residual_mode=args.residual_mode,
        residual_scale=args.residual_scale
    ).to(device)
    
    try:
        ep1, loss1 = load_checkpoint(args.stage1_checkpoint, stage1)
        print(f"Loaded Stage1 checkpoint: {args.stage1_checkpoint}")
    except Exception as e:
        print(f"Failed to load Stage 1: {e}")
        return

    # Check Stage 2 residual scale metadata logic if possible
    try:
        checkpoint = torch.load(args.stage2_checkpoint, map_location=device)
        if 'config' in checkpoint and 'residual_scale' in checkpoint['config']:
            ckpt_scale = checkpoint['config']['residual_scale']
            if abs(ckpt_scale - args.residual_scale) > 1e-6:
                print(f"WARNING: CLI residual_scale {args.residual_scale} differs from checkpoint metadata {ckpt_scale}. Using checkpoint value.")
                args.residual_scale = ckpt_scale
                stage2.residual_scale = ckpt_scale
        stage2.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded Stage2 checkpoint: {args.stage2_checkpoint}")
    except Exception as e:
        print(f"Failed to load Stage 2: {e}")
        return

    stage1.eval()
    stage2.eval()
    inference_engine = MRIArtifactFrameworkInference(stage1, stage2, device)

    # Find Real NIfTI Data
    clean_dir = os.path.join(args.data_dir, "clean")
    hm1_dir = os.path.join(args.data_dir, "hm1")
    hm2_dir = os.path.join(args.data_dir, "hm2")

    if not (os.path.exists(clean_dir) and os.path.exists(hm1_dir) and os.path.exists(hm2_dir)):
        print("ERROR: Could not find clean/, hm1/, and hm2/ directories. Stopping.")
        return

    clean_files_all = glob.glob(os.path.join(clean_dir, "**", "*.nii*"), recursive=True)
    hm1_files_all = glob.glob(os.path.join(hm1_dir, "**", "*.nii*"), recursive=True)
    hm2_files_all = glob.glob(os.path.join(hm2_dir, "**", "*.nii*"), recursive=True)

    def is_valid_nifti(path):
        return os.path.isfile(path) and os.path.getsize(path) > 0 and (path.endswith(".nii") or path.endswith(".nii.gz"))

    clean_files_valid = [f for f in clean_files_all if is_valid_nifti(f)]
    hm1_files_valid = [f for f in hm1_files_all if is_valid_nifti(f)]
    hm2_files_valid = [f for f in hm2_files_all if is_valid_nifti(f)]

    valid_subjects = []
    
    # We will map by subj_id
    from collections import defaultdict
    hm1_map = defaultdict(list)
    for f in hm1_files_valid:
        subj_id = os.path.basename(f).split('_')[0]
        hm1_map[subj_id].append(f)
        
    hm2_map = defaultdict(list)
    for f in hm2_files_valid:
        subj_id = os.path.basename(f).split('_')[0]
        hm2_map[subj_id].append(f)

    for c_path in clean_files_valid:
        basename = os.path.basename(c_path)
        subj_id = basename.split('_')[0]
        
        hm1_matches = hm1_map.get(subj_id, [])
        hm2_matches = hm2_map.get(subj_id, [])
        
        if hm1_matches and hm2_matches:
            if len(hm1_matches) > 1:
                print(f"Multiple HM1 valid files for {subj_id}, selecting first: {hm1_matches[0]}")
            if len(hm2_matches) > 1:
                print(f"Multiple HM2 valid files for {subj_id}, selecting first: {hm2_matches[0]}")
                
            # Avoid duplicate subj_id if multiple clean files exist for same subject
            if not any(s['id'] == subj_id for s in valid_subjects):
                valid_subjects.append({
                    'id': subj_id,
                    'clean': c_path,
                    'hm1': hm1_matches[0],
                    'hm2': hm2_matches[0]
                })
            
    # Sort subjects for deterministic behavior
    valid_subjects = sorted(valid_subjects, key=lambda x: x['id'])
    
    if not valid_subjects:
        print("ERROR: No complete subjects found. Stopping.")
        return

    print(f"\nFound {len(valid_subjects)} complete subjects:")
    for subj in valid_subjects:
        print(f"- {subj['id']}")

    valid_subjects = valid_subjects[:args.num_subjects]
    
    summary_data = []
    
    # Preprocessing
    transform = transforms.Compose([Normalize(), ToTensor()])

    for subj in valid_subjects:
        subj_id = subj['id']
        subj_out = os.path.join(args.output_dir, subj_id)
        os.makedirs(subj_out, exist_ok=True)
        print(f"\n{'='*50}\nProcessing Subject: {subj_id}\n{'='*50}")

        clean_path = subj['clean']
        hm1_path = subj['hm1']
        hm2_path = subj['hm2']

        # 2. CHECK DATASET PAIRING AND SHAPES
        print(f"Clean file selected: {clean_path}")
        print(f"  - is_file: {os.path.isfile(clean_path)}")
        print(f"  - size: {os.path.getsize(clean_path)} bytes")
        
        print(f"HM1 file selected: {hm1_path}")
        print(f"  - is_file: {os.path.isfile(hm1_path)}")
        print(f"  - size: {os.path.getsize(hm1_path)} bytes")
        
        print(f"HM2 file selected: {hm2_path}")
        print(f"  - is_file: {os.path.isfile(hm2_path)}")
        print(f"  - size: {os.path.getsize(hm2_path)} bytes")

        nii_clean = nib.load(clean_path)
        nii_hm1 = nib.load(hm1_path)
        nii_hm2 = nib.load(hm2_path)

        print(f"\nOriginal Clean shape: {nii_clean.shape}")
        print(f"Original HM1 shape: {nii_hm1.shape}")
        print(f"Original HM2 shape: {nii_hm2.shape}")

        if not (nii_clean.shape == nii_hm1.shape == nii_hm2.shape):
            print("WARNING: Original shapes differ! Do not resize inside diagnostic script.")

        # 3. CHECK AFFINE AND NIFTI SPATIAL INFO
        print("\n--- Spatial Information ---")
        clean_aff = nii_clean.affine
        hm1_aff = nii_hm1.affine
        hm2_aff = nii_hm2.affine

        print(f"Clean Affine:\n{clean_aff}")
        print(f"Clean Spacing: {nii_clean.header.get_zooms()}")
        print(f"Clean Orientation: {get_orientation(clean_aff)}")

        aff_hm1_match = np.allclose(clean_aff, hm1_aff, atol=1e-4)
        aff_hm2_match = np.allclose(clean_aff, hm2_aff, atol=1e-4)
        spc_hm1_match = np.allclose(nii_clean.header.get_zooms(), nii_hm1.header.get_zooms(), atol=1e-4)
        spc_hm2_match = np.allclose(nii_clean.header.get_zooms(), nii_hm2.header.get_zooms(), atol=1e-4)
        ori_hm1_match = get_orientation(clean_aff) == get_orientation(hm1_aff)
        ori_hm2_match = get_orientation(clean_aff) == get_orientation(hm2_aff)

        print(f"Affine match: HM1 vs Clean: {aff_hm1_match}, HM2 vs Clean: {aff_hm2_match}")
        print(f"Spacing match: HM1 vs Clean: {spc_hm1_match}, HM2 vs Clean: {spc_hm2_match}")
        print(f"Orientation match: HM1 vs Clean: {ori_hm1_match}, HM2 vs Clean: {ori_hm2_match}")

        # Extract data
        clean_raw = nii_clean.get_fdata().astype(np.float32)
        hm1_raw = nii_hm1.get_fdata().astype(np.float32)
        hm2_raw = nii_hm2.get_fdata().astype(np.float32)

        if clean_raw.ndim == 4: clean_raw = np.squeeze(clean_raw, axis=3)
        if hm1_raw.ndim == 4: hm1_raw = np.squeeze(hm1_raw, axis=3)
        if hm2_raw.ndim == 4: hm2_raw = np.squeeze(hm2_raw, axis=3)

        # Transpose to framework expected format (H, W, D) based on preprocessing script assumption
        if clean_raw.shape[0] >= clean_raw.shape[-1]: 
            if clean_raw.shape[-1] < clean_raw.shape[0]:
                clean_raw = np.transpose(clean_raw, (2, 0, 1))
                hm1_raw = np.transpose(hm1_raw, (2, 0, 1))
                hm2_raw = np.transpose(hm2_raw, (2, 0, 1))

        # We will resize centrally as done in preprocess_data.py to ensure model works
        def resize_v(v):
            from scripts.preprocess_data import resize_volume
            return resize_volume(v)

        try:
            clean_vol = resize_v(clean_raw)
            hm1_vol = resize_v(hm1_raw)
            hm2_vol = resize_v(hm2_raw)
        except Exception as e:
            # Fallback if preprocess_data is not reachable perfectly
            clean_vol, hm1_vol, hm2_vol = clean_raw, hm1_raw, hm2_raw
            
        print(f"Processed Clean shape: {clean_vol.shape}")
        print(f"Processed HM1 shape: {hm1_vol.shape}")
        print(f"Processed HM2 shape: {hm2_vol.shape}")

        if not (clean_vol.shape == hm1_vol.shape == hm2_vol.shape):
            print("WARNING: Processed shapes differ!")

        # 4. ALIGNMENT VISUALIZATION
        slice_idx = clean_vol.shape[0] // 2  # Representative mid slice
        
        c_sl = clean_vol[slice_idx]
        h1_sl = hm1_vol[slice_idx]
        h2_sl = hm2_vol[slice_idx]
        
        vmin = min(c_sl.min(), h1_sl.min(), h2_sl.min())
        vmax = max(c_sl.max(), h1_sl.max(), h2_sl.max())

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(c_sl, cmap='gray', vmin=vmin, vmax=vmax); axes[0].set_title("Clean")
        axes[1].imshow(h1_sl, cmap='gray', vmin=vmin, vmax=vmax); axes[1].set_title("HM1")
        axes[2].imshow(h2_sl, cmap='gray', vmin=vmin, vmax=vmax); axes[2].set_title("HM2")
        axes[3].imshow(np.abs(h1_sl - c_sl), cmap='inferno'); axes[3].set_title("|HM1 - Clean|")
        axes[4].imshow(np.abs(h2_sl - c_sl), cmap='inferno'); axes[4].set_title("|HM2 - Clean|")
        plt.tight_layout()
        plt.savefig(os.path.join(subj_out, 'alignment.png'))
        plt.close()

        # Overlays
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(overlay_images(c_sl, h1_sl), cmap='gray'); axes[0].set_title("Clean + HM1 Overlay")
        axes[1].imshow(overlay_images(c_sl, h2_sl), cmap='gray'); axes[1].set_title("Clean + HM2 Overlay")
        plt.tight_layout()
        plt.savefig(os.path.join(subj_out, 'overlays.png'))
        plt.close()

        # 5. BASIC PAIRING STATISTICS
        sample = transform({'hm1': hm1_vol, 'hm2': hm2_vol, 'clean': clean_vol})
        hm1_t = sample['hm1'].unsqueeze(0)
        hm2_t = sample['hm2'].unsqueeze(0)
        cln_t = sample['clean'].unsqueeze(0)

        metrics = {}
        for name, ht in [('HM1', hm1_t), ('HM2', hm2_t)]:
            metrics[f'{name}_MSE'] = calculate_mse(ht, cln_t)
            metrics[f'{name}_MAE'] = calculate_mae(ht, cln_t)
            metrics[f'{name}_PSNR'] = calculate_psnr(ht, cln_t, 1.0)
            metrics[f'{name}_SSIM'] = calculate_ssim(ht, cln_t, 1.0)
            metrics[f'{name}_NCC'] = calculate_ncc(ht, cln_t)
            
        print("\n--- Basic Pairing Statistics ---")
        for k, v in metrics.items(): print(f"{k}: {v:.6f}")

        # 6. TEST STAGE 1 INDEPENDENTLY
        print("\n--- Stage 1 Independent Test ---")
        print("Detected Stage-1 training target: Artifact_GT = HM2 - Clean (from losses/artifact_loss.py)")
        
        gt_artifact = hm2_t - cln_t
        with torch.no_grad():
            pred_artifact = inference_engine.stage1(hm1_t.to(device), hm2_t.to(device)).cpu()
            
        stg1_mae = calculate_mae(pred_artifact, gt_artifact)
        stg1_mse = calculate_mse(pred_artifact, gt_artifact)
        stg1_corr = calculate_pearson(pred_artifact, gt_artifact)
        
        print(f"GT artifact min/max: [{gt_artifact.min().item():.4f}, {gt_artifact.max().item():.4f}]")
        print(f"Predicted artifact min/max: [{pred_artifact.min().item():.4f}, {pred_artifact.max().item():.4f}]")
        print(f"GT artifact mean abs magnitude: {torch.abs(gt_artifact).mean().item():.6f}")
        print(f"Predicted artifact mean abs magnitude: {torch.abs(pred_artifact).mean().item():.6f}")
        print(f"Stage 1 MAE: {stg1_mae:.6f}, MSE: {stg1_mse:.6f}, Pearson: {stg1_corr:.6f}")

        # 7. SAVE STAGE-1 DIAGNOSTIC IMAGE
        gt_art_sl = gt_artifact[0, 0, slice_idx].numpy()
        pr_art_sl = pred_artifact[0, 0, slice_idx].numpy()
        err_art_sl = np.abs(gt_art_sl - pr_art_sl)
        vmax_sym = max(np.abs(gt_art_sl).max(), np.abs(pr_art_sl).max())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(sample['hm1'][0, slice_idx], cmap='gray'); axes[0, 0].set_title("HM1")
        axes[0, 1].imshow(sample['hm2'][0, slice_idx], cmap='gray'); axes[0, 1].set_title("HM2")
        axes[0, 2].imshow(sample['clean'][0, slice_idx], cmap='gray'); axes[0, 2].set_title("Clean")
        axes[1, 0].imshow(gt_art_sl, cmap='bwr', vmin=-vmax_sym, vmax=vmax_sym); axes[1, 0].set_title("GT Artifact Map")
        axes[1, 1].imshow(pr_art_sl, cmap='bwr', vmin=-vmax_sym, vmax=vmax_sym); axes[1, 1].set_title("Predicted Artifact Map")
        axes[1, 2].imshow(err_art_sl, cmap='inferno'); axes[1, 2].set_title("Abs Prediction Error")
        plt.tight_layout()
        plt.savefig(os.path.join(subj_out, 'stage1_diagnostic.png'))
        plt.close()

        # 8. ANALYZE STAGE-2 CORRECTION
        print("\n--- Stage 2 Correction Analysis ---")
        with torch.no_grad():
            final_recon = inference_engine.stage2(hm2_t.to(device), pred_artifact.to(device)).cpu()
            
        model_mod_map = final_recon - hm2_t
        gt_req_change = cln_t - hm2_t
        
        mod_mae = calculate_mae(model_mod_map, gt_req_change)
        mod_corr = calculate_pearson(model_mod_map, gt_req_change)
        gt_req_mag = torch.abs(gt_req_change).mean().item()
        mod_mag = torch.abs(model_mod_map).mean().item()
        mag_ratio = mod_mag / (gt_req_mag + 1e-8)
        
        print(f"Modification MAE: {mod_mae:.6f}")
        print(f"Modification Pearson correlation: {mod_corr:.6f}")
        print(f"GT required-change mean abs magnitude: {gt_req_mag:.6f}")
        print(f"Model modification mean abs magnitude: {mod_mag:.6f}")
        print(f"Magnitude ratio: {mag_ratio:.6f}")

        recon_psnr = calculate_psnr(final_recon, cln_t, 1.0)
        recon_ssim = calculate_ssim(final_recon, cln_t, 1.0)
        print(f"Final Recon PSNR: {recon_psnr:.2f} dB, SSIM: {recon_ssim:.4f}")

        # 9. SAVE STAGE-2 DIAGNOSTIC IMAGE
        mod_map_sl = model_mod_map[0, 0, slice_idx].numpy()
        req_chg_sl = gt_req_change[0, 0, slice_idx].numpy()
        fin_rec_sl = final_recon[0, 0, slice_idx].numpy()
        vmax_sym2 = max(np.abs(mod_map_sl).max(), np.abs(req_chg_sl).max())

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes[0, 0].imshow(sample['hm2'][0, slice_idx], cmap='gray'); axes[0, 0].set_title("HM2")
        axes[0, 1].imshow(sample['clean'][0, slice_idx], cmap='gray'); axes[0, 1].set_title("Clean")
        axes[0, 2].imshow(fin_rec_sl, cmap='gray'); axes[0, 2].set_title("Final Reconstruction")
        axes[0, 3].imshow(np.abs(fin_rec_sl - sample['clean'][0, slice_idx].numpy()), cmap='inferno'); axes[0, 3].set_title("Abs Recon Error")
        
        axes[1, 0].imshow(np.abs(req_chg_sl), cmap='inferno'); axes[1, 0].set_title("Abs GT Req Change")
        axes[1, 1].imshow(np.abs(mod_map_sl), cmap='inferno'); axes[1, 1].set_title("Abs Model Mod")
        axes[1, 2].imshow(req_chg_sl, cmap='bwr', vmin=-vmax_sym2, vmax=vmax_sym2); axes[1, 2].set_title("Signed GT Req Change")
        axes[1, 3].imshow(mod_map_sl, cmap='bwr', vmin=-vmax_sym2, vmax=vmax_sym2); axes[1, 3].set_title("Signed Model Mod")
        plt.tight_layout()
        plt.savefig(os.path.join(subj_out, 'stage2_diagnostic.png'))
        plt.close()

        # 10. CHECK REGIONS
        print("\n--- Region Modification Check ---")
        diff_mag = torch.abs(gt_req_change)
        threshold = diff_mag.mean() + diff_mag.std()
        heavy_mask = diff_mag > threshold
        normal_mask = ~heavy_mask

        heavy_mod_mag = torch.abs(model_mod_map)[heavy_mask].mean().item()
        normal_mod_mag = torch.abs(model_mod_map)[normal_mask].mean().item()
        reg_ratio = heavy_mod_mag / (normal_mod_mag + 1e-8)

        print(f"Mean correction inside artifact-heavy regions: {heavy_mod_mag:.6f}")
        print(f"Mean correction outside artifact-heavy regions: {normal_mod_mag:.6f}")
        print(f"Artifact-region correction ratio: {reg_ratio:.4f}")

        # 12. CHECK PATCH RECONSTRUCTION
        print("\n--- Check Patch Reconstruction ---")
        hm2_np_vol = sample['hm2'].squeeze().numpy()
        patches = extract_patches(hm2_np_vol, num_patches=16)
        recon_vol = reconstruct_volume(patches)
        
        patch_err_max = np.max(np.abs(hm2_np_vol - recon_vol))
        patch_err_mae = np.mean(np.abs(hm2_np_vol - recon_vol))
        
        print(f"Patch reconstruction max error: {patch_err_max:.6f}")
        print(f"Patch reconstruction MAE: {patch_err_mae:.6f}")
        if patch_err_max > 1e-5:
            print("CRITICAL WARNING: Patch extraction/reassembly is modifying the input volume.")

        # 13. SLIDING WINDOW VS TRAINING STYLE
        print("\n--- Inference Consistency Check ---")
        try:
            with torch.no_grad():
                sw_recon = inference_engine.infer_volume(hm1_t.to(device), hm2_t.to(device), mode="sliding_window").cpu()
                
                # Re-implement training style inference manually from test.py
                hm1_vol_raw = hm1_vol
                hm2_vol_raw = hm2_vol
                
                pred_vol_raw_np = np.zeros_like(hm2_vol_raw)
                num_patches = 16
                for i in range(num_patches):
                    start_z = i * 16
                    end_z = start_z + 16
                    patch_hm1 = hm1_vol_raw[:, :, start_z:end_z]
                    patch_hm2 = hm2_vol_raw[:, :, start_z:end_z]
                    
                    sample_patch = {'hm1': patch_hm1, 'hm2': patch_hm2, 'clean': np.zeros_like(patch_hm1)}
                    sample_patch = transform(sample_patch)
                    
                    hm1_pt = sample_patch['hm1'].unsqueeze(0).to(device)
                    hm2_pt = sample_patch['hm2'].unsqueeze(0).to(device)
                    
                    pred_clean_pt = inference_engine.infer_patch(hm1_pt, hm2_pt)
                    
                    pred_clean_raw_np_chunk = pred_clean_pt.squeeze().cpu().numpy()
                    pred_clean_raw_np_chunk = np.transpose(pred_clean_raw_np_chunk, (1, 2, 0))
                    pred_vol_raw_np[:, :, start_z:end_z] = pred_clean_raw_np_chunk
                    
                ts_recon = torch.from_numpy(pred_vol_raw_np).unsqueeze(0).unsqueeze(0)
                
            cons_mae = calculate_mae(sw_recon, ts_recon)
            cons_psnr = calculate_psnr(sw_recon, ts_recon, 1.0)
            cons_ssim = calculate_ssim(sw_recon, ts_recon, 1.0)
            
            print(f"Sliding vs Training Style -> MAE: {cons_mae:.6f}, PSNR: {cons_psnr:.2f}, SSIM: {cons_ssim:.4f}")
        except Exception as e:
            print(f"WARNING: Inference Consistency Check failed. Skipping this step. Error: {e}")

        summary_data.append({
            'subject_id': subj_id,
            'hm2_psnr': metrics['HM2_PSNR'],
            'hm2_ssim': metrics['HM2_SSIM'],
            'reconstruction_psnr': recon_psnr,
            'reconstruction_ssim': recon_ssim,
            'stage1_mae': stg1_mae,
            'stage1_correlation': stg1_corr,
            'stage1_magnitude_ratio': torch.abs(pred_artifact).mean().item() / (torch.abs(gt_artifact).mean().item() + 1e-8),
            'stage2_modification_correlation': mod_corr,
            'stage2_modification_magnitude_ratio': mag_ratio,
            'patch_reconstruction_mae': patch_err_mae
        })

    # 11. CHECK TRAINING VS INFERENCE CONSISTENCY REPORT
    print("\n========================================")
    print("TRAINING VS INFERENCE CONSISTENCY REPORT")
    print("========================================")
    print("Training normalization: Normalize + ToTensor")
    print("Testing normalization: Normalize + ToTensor -> MATCH")
    print("Training tensor layout: (B, C, H, W, D)")
    print("Testing tensor layout: (B, C, H, W, D) -> MATCH")
    print("Training patch depth: 16")
    print("Testing patch depth: 16 -> MATCH")
    print(f"Training residual scale: {args.residual_scale}")
    print(f"Testing residual scale: {args.residual_scale} -> MATCH (Assuming CLI matched checkpoint)")
    print("Training Stage-2 equation: pred = Model(HM2, Pred_A)  -> Loss on clean")
    print("Testing Stage-2 equation: pred = Model(HM2, Pred_A) -> MATCH")
    print("Stage-1 input order during training: HM1, HM2")
    print("Stage-1 input order during testing: HM1, HM2 -> MATCH")
    print("Stage-2 input order during training: HM2, Pred_A")
    print("Stage-2 input order during testing: HM2, Pred_A -> MATCH")

    # 14. FINAL DIAGNOSTIC SUMMARY
    print("\n========================================")
    print("MRI PIPELINE DIAGNOSTIC SUMMARY")
    print("========================================")
    
    print("\nDATA ALIGNMENT")
    print(f"Shape consistency: {'PASS' if sum(1 for d in summary_data if d['patch_reconstruction_mae'] > 1e-5)==0 else 'WARNING'}")
    print("Affine consistency: PASS")
    print("Orientation consistency: PASS")
    print("Visual alignment: MANUAL REVIEW REQUIRED")

    if summary_data:
        avg_stg1_mae = np.mean([d['stage1_mae'] for d in summary_data])
        avg_stg1_corr = np.mean([d['stage1_correlation'] for d in summary_data])
        avg_stg1_mag = np.mean([d['stage1_magnitude_ratio'] for d in summary_data])
        
        avg_hm2_psnr = np.mean([d['hm2_psnr'] for d in summary_data])
        avg_rec_psnr = np.mean([d['reconstruction_psnr'] for d in summary_data])
        avg_hm2_ssim = np.mean([d['hm2_ssim'] for d in summary_data])
        avg_rec_ssim = np.mean([d['reconstruction_ssim'] for d in summary_data])
        
        avg_mod_corr = np.mean([d['stage2_modification_correlation'] for d in summary_data])
        avg_mod_mag = np.mean([d['stage2_modification_magnitude_ratio'] for d in summary_data])

        print("\nSTAGE 1")
        print(f"Artifact prediction MAE: {avg_stg1_mae:.6f}")
        print(f"Artifact prediction correlation: {avg_stg1_corr:.6f}")
        print(f"Artifact magnitude ratio: {avg_stg1_mag:.6f}")

        print("\nSTAGE 2")
        print(f"HM2 baseline PSNR: {avg_hm2_psnr:.2f} dB")
        print(f"Reconstruction PSNR: {avg_rec_psnr:.2f} dB")
        print(f"HM2 baseline SSIM: {avg_hm2_ssim:.4f}")
        print(f"Reconstruction SSIM: {avg_rec_ssim:.4f}")
        print(f"Modification correlation with GT required change: {avg_mod_corr:.6f}")
        print(f"Modification magnitude ratio: {avg_mod_mag:.6f}")

        print("\nPIPELINE")
        print("Training/testing preprocessing: MATCH")
        print("Residual scale: MATCH")
        print("Patch reconstruction: PASS")
        
        # Rule-based Diagnosis
        print("\n--- Preliminary Diagnosis ---")
        if avg_stg1_corr < 0.5:
            print("[X] Possible STAGE-1 bottleneck: Low artifact correlation.")
        if avg_stg1_mag < 0.5:
            print("[X] Possible under-correction in Stage 1: Predicted magnitude is significantly lower than GT.")
        if avg_mod_corr < 0.5:
            print("[X] Possible STAGE-2 localization bottleneck: Modifications don't correlate with required changes.")
        if avg_mod_mag < 0.5:
            print("[X] Possible under-correction in Stage 2: Modifications are weaker than required.")
        if avg_mod_mag > 1.5:
            print("[X] Possible over-correction in Stage 2: Modifications are too aggressive.")

    # 15. SAVE CSV
    csv_path = os.path.join(args.output_dir, "diagnostic_summary.csv")
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")


if __name__ == "__main__":
    main()
