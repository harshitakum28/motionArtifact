import os
import sys
import torch
import numpy as np
import argparse
import glob
import math
import matplotlib.pyplot as plt
from torchvision import transforms

# Insert specific dataset directory first to avoid shadowing by Hugging Face `datasets`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DATASETS_DIR)

from configs.config import Config
from models.stage1 import Stage1ArtifactPredictor
from models.stage2 import Stage2ArtifactGuidedReconstructor
from inference.inference import MRIArtifactFrameworkInference
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mse
from utils.checkpoint import load_checkpoint_strict_for_inference
from datasets.transforms import ToTensor

def normalize_full_volume(vol):
    """
    Perform preprocessing / percentile clipping / normalization on the COMPLETE full volume.
    """
    perc_low = float(np.percentile(vol, 0.5))
    perc_high = float(np.percentile(vol, 99.5))
    perc_clipped = np.clip(vol, perc_low, perc_high)
    return (perc_clipped - perc_low) / (perc_high - perc_low + 1e-8)

def global_min_max_normalize(vol):
    """
    Global min-max normalization for the full reconstructed volume.
    """
    raw_min = vol.min()
    raw_max = vol.max()
    return (vol - raw_min) / (raw_max - raw_min + 1e-8)

def evaluate_volume(pred_vol, clean_vol):
    metrics = {'psnr': [], 'ssim': [], 'mse': []}
    
    d = clean_vol.shape[2]
    inf_count = 0
    for z in range(d):
        gt_slice = clean_vol[:, :, z]
        pred_slice = pred_vol[:, :, z]
        
        gt_t = torch.from_numpy(gt_slice).unsqueeze(0).unsqueeze(0).float()
        pred_t = torch.from_numpy(pred_slice).unsqueeze(0).unsqueeze(0).float()
        
        p_mod = calculate_psnr(pred_t, gt_t, data_range=1.0)
        if math.isinf(p_mod):
            inf_count += 1
        else:
            metrics['psnr'].append(p_mod)
            
        metrics['mse'].append(calculate_mse(pred_t, gt_t))
        metrics['ssim'].append(calculate_ssim(pred_t, gt_t, data_range=1.0))
        
    def mean(lst): return sum(lst)/len(lst) if lst else 0.0
    
    return {
        'psnr': mean(metrics['psnr']),
        'ssim': mean(metrics['ssim']),
        'mse': mean(metrics['mse'])
    }

def run_transform_identity_check():
    dummy_np = np.random.rand(192, 256, 16).astype(np.float32)
    transform = ToTensor()
    sample = transform({'dummy': dummy_np})
    tensor_dummy = sample['dummy'].unsqueeze(0)
    assert tensor_dummy.shape == (1, 1, 16, 192, 256)
    back_np = tensor_dummy.squeeze(0).squeeze(0).numpy().transpose((1, 2, 0))
    assert back_np.shape == (192, 256, 16)
    max_err = np.max(np.abs(dummy_np - back_np))
    assert max_err == 0.0
    print("NumPy -> Tensor -> NumPy round trip MAX ERROR = 0.0. Test PASSED.")

def run_split_concat_identity_test(vol_clean, vol_hm1, vol_hm2):
    for name, vol in [("Clean", vol_clean), ("HM1", vol_hm1), ("HM2", vol_hm2)]:
        patches = []
        for i in range(16):
            start_z = i * 16
            end_z = start_z + 16
            patch = vol[:, :, start_z:end_z]
            assert patch.shape == (192, 256, 16)
            patches.append(patch)
        assert len(patches) == 16
        reconstructed = np.concatenate(patches, axis=2)
        assert reconstructed.shape == (192, 256, 256)
        max_err = np.max(np.abs(vol - reconstructed))
        assert max_err == 0.0
    print("Split -> Concatenate Patch Identity Test PASSED.")

def main():
    parser = argparse.ArgumentParser(description="Test MRI Artifact Removal Framework")
    parser.add_argument("--stage1_checkpoint", type=str, required=True, help="Path to pre-trained Stage 1 checkpoint")
    parser.add_argument("--stage2_checkpoints", nargs='+', required=True, help="Path(s) to Stage 2 checkpoint(s)")
    parser.add_argument("--data_dir", type=str, default="", help="Directory for testing data. Uses Config default if empty.")
    parser.add_argument("--eval_mode", type=str, choices=['full_volume', 'patch_based'], required=True, help="Evaluation mode (full_volume or patch_based)")
    args = parser.parse_args()

    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # DIRECTORY AUDIT & RESOLUTION
    # ---------------------------------------------------------
    if args.data_dir:
        resolved_data_dir = args.data_dir
    elif hasattr(config, 'TEST_DATA_DIR') and config.TEST_DATA_DIR:
        resolved_data_dir = config.TEST_DATA_DIR
    else:
        resolved_data_dir = config.DATA_DIR
        
    print("\n========================================")
    print("PRE-EXECUTION DIRECTORY AUDIT")
    print("========================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Resolved dataset directory: {resolved_data_dir}")
    print(f"Stage-1 checkpoint path: {args.stage1_checkpoint}")
    print(f"Number of Stage-2 checkpoints: {len(args.stage2_checkpoints)}")
    for p in args.stage2_checkpoints:
        print(f"  - {p}")
        
    if not os.path.exists(resolved_data_dir):
        print("\nERROR: Dataset directory does not exist.")
        print(f"Resolved directory: {resolved_data_dir}")
        sys.exit(1)
        
    sub_dirs = sorted(glob.glob(os.path.join(resolved_data_dir, "sub-*"))) + \
               sorted(glob.glob(os.path.join(resolved_data_dir, "subject_*")))
               
    if not sub_dirs:
        print(f"\nERROR: No subjects found in {resolved_data_dir}")
        print("Immediate directory contents:")
        for item in os.listdir(resolved_data_dir):
            print(f"  - {item}")
        print("Expected subject naming patterns: 'sub-*' or 'subject_*'")
        sys.exit(1)
        
    subjects_to_process = []
    for subj_path in sub_dirs:
        if os.path.exists(os.path.join(subj_path, "hm1.npy")):
            subjects_to_process.append((os.path.basename(subj_path), subj_path))
            
    if not subjects_to_process:
        print(f"\nERROR: Subjects found, but missing required .npy files (hm1.npy).")
        sys.exit(1)
        
    print(f"Number of discovered subjects: {len(subjects_to_process)}")
    print(f"Discovered subject names: {[s[0] for s in subjects_to_process]}")
    print(f"Evaluation Mode: {args.eval_mode}")
    print("========================================\n")
    
    # Validation of checkpoints existence
    if not os.path.isfile(args.stage1_checkpoint):
        raise FileNotFoundError(f"Stage 1 checkpoint missing at: {args.stage1_checkpoint}")
    for p in args.stage2_checkpoints:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Stage 2 checkpoint missing at: {p}")

    # Generate unique output IDs and check for collisions
    unique_ids = set()
    checkpoint_id_map = {}
    for p in args.stage2_checkpoints:
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(p)))
        filename_no_ext = os.path.splitext(os.path.basename(p))[0]
        unique_id = f"{parent_dir}__{filename_no_ext}"
        
        # Disambiguate if needed
        counter = 1
        original_id = unique_id
        while unique_id in unique_ids:
            unique_id = f"{original_id}_{counter}"
            counter += 1
            
        unique_ids.add(unique_id)
        checkpoint_id_map[p] = unique_id
        
    run_transform_identity_check()
    
    # ---------------------------------------------------------
    # PRE-LOAD FULL VOLUMES (Already Normalized in Preprocessing)
    # ---------------------------------------------------------
    normalized_volumes = {}
    print("\n======================================")
    print("LOADING PRE-NORMALIZED FULL VOLUMES")
    print("======================================")
    for subj_name, subj_path in subjects_to_process:
        print(f"Loading {subj_name} (pre-normalized)...")
        hm1_vol_norm = np.load(os.path.join(subj_path, "hm1.npy"))
        hm2_vol_norm = np.load(os.path.join(subj_path, "hm2.npy"))
        gt_clean_norm = np.load(os.path.join(subj_path, "clean.npy"))
        
        normalized_volumes[subj_name] = {
            'hm1': hm1_vol_norm,
            'hm2': hm2_vol_norm,
            'clean': gt_clean_norm,
            'hm2_raw': hm2_vol_norm, # keep just in case
        }
    
    # ---------------------------------------------------------
    # LOAD STAGE 1
    # ---------------------------------------------------------
    stage1 = Stage1ArtifactPredictor(base_filters=config.BASE_FILTERS_STAGE1).to(device)
    load_checkpoint_strict_for_inference(args.stage1_checkpoint, stage1, device)
    stage1.eval()
    
    inference_engine = MRIArtifactFrameworkInference(stage1, stage1, device) # dummy stage2 for now
    
    # ---------------------------------------------------------
    # STAGE 1 CACHING
    # ---------------------------------------------------------
    print("\n======================================")
    print("PRE-CALCULATING STAGE-1 OUTPUTS (CACHING)")
    print("======================================")
    
    stage1_cache = {}
    transform = transforms.Compose([ToTensor()])
    
    for subj_name, _ in subjects_to_process:
        print(f"Caching Stage-1 for {subj_name} in {args.eval_mode} mode...")
        hm1_norm = normalized_volumes[subj_name]['hm1']
        hm2_norm = normalized_volumes[subj_name]['hm2']
        
        if args.eval_mode == 'patch_based':
            subject_cache = []
            for i in range(16):
                start_z = i * 16
                end_z = start_z + 16
                patch_hm1 = hm1_norm[:, :, start_z:end_z]
                patch_hm2 = hm2_norm[:, :, start_z:end_z]
                
                sample_patch = transform({'hm1': patch_hm1, 'hm2': patch_hm2})
                hm1_pt = sample_patch['hm1'].unsqueeze(0).to(device)
                hm2_pt = sample_patch['hm2'].unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred_artifact = inference_engine.infer_stage1_patch(hm1_pt, hm2_pt)
                    # Verification Test
                    pred_artifact_fresh = inference_engine.infer_stage1_patch(hm1_pt, hm2_pt)
                    max_diff = torch.max(torch.abs(pred_artifact - pred_artifact_fresh)).item()
                    assert max_diff <= 1e-6, f"Stage-1 caching not numerically identical! Diff: {max_diff}"
                    
                subject_cache.append({'hm2_pt': hm2_pt, 'pred_artifact': pred_artifact})
            stage1_cache[subj_name] = subject_cache
            
        elif args.eval_mode == 'full_volume':
            sample_full = transform({'hm1': hm1_norm, 'hm2': hm2_norm})
            hm1_pt = sample_full['hm1'].unsqueeze(0).to(device)
            hm2_pt = sample_full['hm2'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_artifact = inference_engine.infer_stage1_patch(hm1_pt, hm2_pt)
                # Verification Test
                pred_artifact_fresh = inference_engine.infer_stage1_patch(hm1_pt, hm2_pt)
                max_diff = torch.max(torch.abs(pred_artifact - pred_artifact_fresh)).item()
                assert max_diff <= 1e-6, f"Stage-1 caching not numerically identical! Diff: {max_diff}"
                
            stage1_cache[subj_name] = {'hm2_pt': hm2_pt, 'pred_artifact': pred_artifact}
        
    # ---------------------------------------------------------
    # STAGE 2 EVALUATION
    # ---------------------------------------------------------
    checkpoint_results = {}
    
    for ckpt_path in args.stage2_checkpoints:
        ckpt_id = checkpoint_id_map[ckpt_path]
        print(f"\n======================================")
        print(f"EVALUATING CHECKPOINT: {ckpt_path}")
        print(f"UNIQUE EXPERIMENT ID: {ckpt_id}")
        print(f"======================================")
        
        stage2 = Stage2ArtifactGuidedReconstructor(
            base_filters=config.BASE_FILTERS_STAGE2,
            num_residual_blocks=config.NUM_RESIDUAL_BLOCKS
        ).to(device)
        
        # Robust Strict Loading
        load_checkpoint_strict_for_inference(ckpt_path, stage2, device)
        stage2.eval()
        inference_engine.stage2 = stage2
            
        ckpt_metrics = {'hm2': {'psnr':[], 'ssim':[], 'mse':[]},
                        'raw': {'psnr':[], 'ssim':[], 'mse':[]},
                        'norm': {'psnr':[], 'ssim':[], 'mse':[]}}
                        
        for subj_name, _ in subjects_to_process:
            print(f"\n--- Subject: {subj_name} ---")
            
            # Use pre-normalized full volumes
            hm1_norm = normalized_volumes[subj_name]['hm1']
            hm2_norm = normalized_volumes[subj_name]['hm2']
            clean_norm = normalized_volumes[subj_name]['clean']
            
            # Identity test on normalized volumes
            run_split_concat_identity_test(clean_norm, hm1_norm, hm2_norm)
            
            if args.eval_mode == 'patch_based':
                pred_vol_raw_np = np.zeros_like(clean_norm)
                subject_cache = stage1_cache[subj_name]
                
                for i in range(16):
                    start_z = i * 16
                    end_z = start_z + 16
                    hm2_pt = subject_cache[i]['hm2_pt']
                    pred_artifact = subject_cache[i]['pred_artifact']
                    
                    with torch.no_grad():
                        pred_clean_pt = inference_engine.infer_stage2_patch(hm2_pt, pred_artifact)
                        
                    # Store RAW reconstructed patch exactly - NO patch-level normalization allowed
                    pred_clean_raw_np_patch = pred_clean_pt.squeeze().cpu().numpy().transpose((1, 2, 0))
                    pred_vol_raw_np[:, :, start_z:end_z] = pred_clean_raw_np_patch
                    
            elif args.eval_mode == 'full_volume':
                subject_cache = stage1_cache[subj_name]
                hm2_pt = subject_cache['hm2_pt']
                pred_artifact = subject_cache['pred_artifact']
                
                with torch.no_grad():
                    pred_clean_pt = inference_engine.infer_stage2_patch(hm2_pt, pred_artifact)
                    
                pred_vol_raw_np = pred_clean_pt.squeeze().cpu().numpy().transpose((1, 2, 0))
                
            assert pred_vol_raw_np.shape == (192, 256, 256)
            
            raw_min = pred_vol_raw_np.min()
            raw_max = pred_vol_raw_np.max()
            print(f"Raw Reconstruction Range: [{raw_min:.4f}, {raw_max:.4f}]")
            
            # To evaluate RAW performance fairly for metrics, we clip it to [0, 1]
            clamped_raw_vol = np.clip(pred_vol_raw_np, 0.0, 1.0)
            
            # Global Min-Max Normalization of the FULL reconstructed volume
            global_normalized_vol = global_min_max_normalize(pred_vol_raw_np)
            
            norm_min = global_normalized_vol.min()
            norm_max = global_normalized_vol.max()
            print(f"Globally Normalized Reconstruction Range: [{norm_min:.4f}, {norm_max:.4f}]")
            
            # Evaluate using Clean Normalized as ground truth
            # We compare HM2 (normalized), RAW (clamped), and Global-Normalized
            res_hm2 = evaluate_volume(hm2_norm, clean_norm)
            res_raw = evaluate_volume(clamped_raw_vol, clean_norm)
            res_norm = evaluate_volume(global_normalized_vol, clean_norm)
            
            for k in ['psnr', 'ssim', 'mse']:
                ckpt_metrics['hm2'][k].append(res_hm2[k])
                ckpt_metrics['raw'][k].append(res_raw[k])
                ckpt_metrics['norm'][k].append(res_norm[k])
                
            # Output saving and Visualization
            vis_dir = os.path.join(config.ROOT_DIR, "output", ckpt_id, subj_name)
            os.makedirs(vis_dir, exist_ok=True)
            
            np.save(os.path.join(vis_dir, "raw_reconstruction.npy"), pred_vol_raw_np)
            np.save(os.path.join(vis_dir, "global_normalized_reconstruction.npy"), global_normalized_vol)
            
            slice_idx = 128
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            axes[0].imshow(hm1_norm[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[0].set_title("HM1 (Norm)")
            axes[0].axis('off')
            
            axes[1].imshow(hm2_norm[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[1].set_title("HM2 (Norm)")
            axes[1].axis('off')
            
            axes[2].imshow(clean_norm[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[2].set_title("Clean (Norm)")
            axes[2].axis('off')
            
            axes[3].imshow(pred_vol_raw_np[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[3].set_title("RAW Reconstruction")
            axes[3].axis('off')
            
            axes[4].imshow(global_normalized_vol[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[4].set_title("Globally Min-Max Normalized")
            axes[4].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "comparison.png"))
            plt.close()
            
        checkpoint_results[ckpt_id] = ckpt_metrics
        
        # Summary for this checkpoint
        def m(lst): return np.mean(lst) if lst else 0
        print(f"\n======================================")
        print(f"CHECKPOINT SUMMARY: {ckpt_id}")
        print(f"======================================")
        print("Average HM2 Baseline:")
        print(f"PSNR: {m(ckpt_metrics['hm2']['psnr']):.2f}")
        print(f"SSIM: {m(ckpt_metrics['hm2']['ssim']):.4f}")
        print(f"MSE:  {m(ckpt_metrics['hm2']['mse']):.6f}")
        print("\nAverage RAW Reconstruction:")
        print(f"PSNR: {m(ckpt_metrics['raw']['psnr']):.2f}")
        print(f"SSIM: {m(ckpt_metrics['raw']['ssim']):.4f}")
        print(f"MSE:  {m(ckpt_metrics['raw']['mse']):.6f}")
        print("\nAverage Globally Normalized Reconstruction:")
        print(f"PSNR: {m(ckpt_metrics['norm']['psnr']):.2f}")
        print(f"SSIM: {m(ckpt_metrics['norm']['ssim']):.4f}")
        print(f"MSE:  {m(ckpt_metrics['norm']['mse']):.6f}")
        print(f"======================================\n")
        
    print(f"\n======================================")
    print("FINAL COMPARISON TABLE")
    print(f"======================================")
    print(f"{'Checkpoint ID':<40} | {'Raw PSNR':<10} | {'Raw SSIM':<10} | {'Raw MSE':<10} | {'Global-Norm PSNR':<16} | {'Global-Norm SSIM':<16} | {'Global-Norm MSE':<15}")
    print("-" * 135)
    
    best_raw_psnr = (-1, "")
    best_raw_ssim = (-1, "")
    best_norm_psnr = (-1, "")
    best_norm_ssim = (-1, "")
    
    for ckpt_id, metrics in checkpoint_results.items():
        def m(k1, k2): return np.mean(metrics[k1][k2])
        
        r_psnr = m('raw', 'psnr')
        r_ssim = m('raw', 'ssim')
        r_mse = m('raw', 'mse')
        
        n_psnr = m('norm', 'psnr')
        n_ssim = m('norm', 'ssim')
        n_mse = m('norm', 'mse')
        
        if r_psnr > best_raw_psnr[0]: best_raw_psnr = (r_psnr, ckpt_id)
        if r_ssim > best_raw_ssim[0]: best_raw_ssim = (r_ssim, ckpt_id)
        if n_psnr > best_norm_psnr[0]: best_norm_psnr = (n_psnr, ckpt_id)
        if n_ssim > best_norm_ssim[0]: best_norm_ssim = (n_ssim, ckpt_id)
        
        print(f"{ckpt_id:<40} | {r_psnr:<10.2f} | {r_ssim:<10.4f} | {r_mse:<10.6f} | {n_psnr:<16.2f} | {n_ssim:<16.4f} | {n_mse:<15.6f}")
        
    print(f"\nBest RAW PSNR checkpoint: {best_raw_psnr[1]} ({best_raw_psnr[0]:.2f})")
    print(f"Best RAW SSIM checkpoint: {best_raw_ssim[1]} ({best_raw_ssim[0]:.4f})")
    print(f"Best Global-Normalized PSNR checkpoint: {best_norm_psnr[1]} ({best_norm_psnr[0]:.2f})")
    print(f"Best Global-Normalized SSIM checkpoint: {best_norm_ssim[1]} ({best_norm_ssim[0]:.4f})")

if __name__ == '__main__':
    main()
