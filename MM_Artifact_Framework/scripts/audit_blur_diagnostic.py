import os
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import ndimage
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from configs.config import Config
from models.stage1_artifact_predictor import Stage1ArtifactPredictor
from models.stage2_reconstructor import Stage2ArtifactGuidedReconstructor
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from datasets.transforms import ToTensor
from torchvision import transforms

def compute_masked_gradient_energy(img, mask):
    """Computes Sobel gradient magnitude, masked by foreground."""
    gx = ndimage.sobel(img, axis=0)
    gy = ndimage.sobel(img, axis=1)
    mag = np.sqrt(gx**2 + gy**2)
    masked_mag = mag * mask
    return np.sum(masked_mag**2), masked_mag

def compute_hf_energy(img, mask, cutoff_ratio):
    """Computes High-Frequency Fourier energy using a radial Nyquist cutoff."""
    # Compute FFT natively without visualization normalization
    F = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(F)**2
    
    H, W = img.shape
    Y, X = np.ogrid[-H/2:H/2, -W/2:W/2]
    R = np.sqrt(X**2 + Y**2)
    R_max = min(H, W) / 2.0
    
    hf_mask = R > (cutoff_ratio * R_max)
    hf_energy = np.sum(power[hf_mask])
    total_energy = np.sum(power)
    
    return hf_energy, total_energy

def compute_pearson_corr(arr1, arr2, mask=None):
    if mask is not None:
        return np.corrcoef(arr1[mask].flatten(), arr2[mask].flatten())[0, 1]
    return np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]

def process_subject(subj_path, stage1, stage2, device, output_dir, slices=[64, 128, 192]):
    subj_name = os.path.basename(subj_path.rstrip("/\\"))
    print(f"\n{'='*80}\nProcessing {subj_name}...\n{'='*80}")
    
    clean_vol = np.load(os.path.join(subj_path, "clean.npy"))
    hm1_vol = np.load(os.path.join(subj_path, "hm1.npy"))
    hm2_vol = np.load(os.path.join(subj_path, "hm2.npy"))
    
    # Run Inference in Training-Style patches (Axis 2, 16 patches)
    pred_clean_vol = np.zeros_like(clean_vol)
    pred_artifact_vol = np.zeros_like(clean_vol)
    
    transform = transforms.Compose([ToTensor()])
    
    for i in range(16):
        sz = i * 16
        ez = sz + 16
        
        patch_hm1 = hm1_vol[:, :, sz:ez]
        patch_hm2 = hm2_vol[:, :, sz:ez]
        
        sample = transform({'hm1': patch_hm1, 'hm2': patch_hm2})
        hm1_patch = sample['hm1'].unsqueeze(0).to(device)
        hm2_patch = sample['hm2'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            p_art = stage1(hm1_patch, hm2_patch)
            p_clean = stage2(hm2_patch, p_art)
            
        pred_artifact_vol[:, :, sz:ez] = p_art.squeeze().cpu().numpy().transpose((1, 2, 0))
        pred_clean_vol[:, :, sz:ez] = p_clean.squeeze().cpu().numpy().transpose((1, 2, 0))
        
    pred_clean_vol_clamped = np.clip(pred_clean_vol, 0.0, 1.0)
    
    subject_metrics = []
    
    for sl in slices:
        print(f"\n--- Analyzing Slice {sl} ---")
        c_img = clean_vol[:, :, sl]
        h2_img = hm2_vol[:, :, sl]
        p_img_raw = pred_clean_vol[:, :, sl]
        p_img = pred_clean_vol_clamped[:, :, sl]
        
        gt_art = h2_img - c_img
        p_art = pred_artifact_vol[:, :, sl]
        
        mask = (c_img > 0.05)
        
        print(f"Clean min/max: {c_img.min():.4f} / {c_img.max():.4f}")
        print(f"Recon raw min/max: {p_img_raw.min():.4f} / {p_img_raw.max():.4f}")
        print(f"Recon clamped min/max: {p_img.min():.4f} / {p_img.max():.4f}")
        
        # TASK 1: Gradient Energy Audit
        c_ge_whole, _ = compute_masked_gradient_energy(c_img, np.ones_like(c_img))
        p_ge_whole, _ = compute_masked_gradient_energy(p_img, np.ones_like(p_img))
        
        c_ge_masked, c_grad = compute_masked_gradient_energy(c_img, mask)
        h2_ge_masked, h2_grad = compute_masked_gradient_energy(h2_img, mask)
        p_ge_masked, p_grad = compute_masked_gradient_energy(p_img, mask)
        
        print(f"\nClean GE (whole): {c_ge_whole:.4f} | Recon GE (whole): {p_ge_whole:.4f}")
        print(f"Clean GE (mask):  {c_ge_masked:.4f} | HM2 GE (mask): {h2_ge_masked:.4f} | Recon GE (mask): {p_ge_masked:.4f}")
        print(f"HM2/Clean GE Ratio: {h2_ge_masked/c_ge_masked:.4f} | Recon/Clean GE Ratio: {p_ge_masked/c_ge_masked:.4f}")
        
        # TASK 2: HF Energy Audit
        hf_ratios = {}
        for cutoff in [0.25, 0.50, 0.75]:
            c_hf, c_tot = compute_hf_energy(c_img, mask, cutoff)
            h2_hf, _ = compute_hf_energy(h2_img, mask, cutoff)
            p_hf, _ = compute_hf_energy(p_img, mask, cutoff)
            
            hf_ratios[f'hm2_{cutoff}'] = h2_hf / (c_hf + 1e-8)
            hf_ratios[f'recon_{cutoff}'] = p_hf / (c_hf + 1e-8)
            
            if cutoff == 0.50:
                print(f"\nHF Cutoff 0.50:")
                print(f"Clean HF Power / Total Power: {c_hf/c_tot * 100:.2f}%")
                print(f"HM2 HF Retention: {hf_ratios['hm2_0.5']*100:.2f}%")
                print(f"Recon HF Retention: {hf_ratios['recon_0.5']*100:.2f}%")

        # TASK 3: Artifact Quality and Excess Leakage
        art_mae = np.mean(np.abs(gt_art - p_art))
        art_mse = np.mean((gt_art - p_art)**2)
        art_pearson = compute_pearson_corr(gt_art, p_art, mask)
        
        _, gt_art_grad = compute_masked_gradient_energy(gt_art, mask)
        _, p_art_grad = compute_masked_gradient_energy(p_art, mask)
        
        grad_gt_pred_corr = compute_pearson_corr(gt_art_grad, p_art_grad, mask)
        
        clean_gt_corr = compute_pearson_corr(c_grad, gt_art_grad, mask)
        clean_pred_corr = compute_pearson_corr(c_grad, p_art_grad, mask)
        excess_leakage = clean_pred_corr - clean_gt_corr
        
        print(f"\nArtifact Pearson: {art_pearson:.4f} | Grad Corr: {grad_gt_pred_corr:.4f}")
        print(f"Clean-vs-GT Grad Corr (Natural Leakage): {clean_gt_corr:.4f}")
        print(f"Clean-vs-Pred Grad Corr (Predicted Leakage): {clean_pred_corr:.4f}")
        print(f"EXCESS LEAKAGE: {excess_leakage:.4f}")
        
        # TASK 4: Overall Stage-2 performance
        # We calculate PSNR/SSIM over the masked region if possible, or full image (standard)
        # We use full image to match standard metrics
        h2_psnr = psnr(c_img, h2_img, data_range=1.0)
        h2_ssim = ssim(c_img, h2_img, data_range=1.0)
        p_psnr = psnr(c_img, p_img, data_range=1.0)
        p_ssim = ssim(c_img, p_img, data_range=1.0)
        
        print(f"\nHM2 PSNR: {h2_psnr:.2f} | Recon PSNR: {p_psnr:.2f}")
        print(f"HM2 SSIM: {h2_ssim:.4f} | Recon SSIM: {p_ssim:.4f}")
        
        metrics = {
            "subj": subj_name,
            "slice": sl,
            "recon_ge_ratio": p_ge_masked / c_ge_masked,
            "hm2_ge_ratio": h2_ge_masked / c_ge_masked,
            "recon_hf_050": hf_ratios['recon_0.5'],
            "hm2_hf_050": hf_ratios['hm2_0.5'],
            "art_pearson": art_pearson,
            "excess_leakage": excess_leakage,
            "psnr_diff": p_psnr - h2_psnr
        }
        subject_metrics.append(metrics)
        
        # Plotting
        plt.style.use('dark_background')
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f"Audit Analysis - {subj_name} (Slice {sl})", fontsize=16)
        
        def show(ax, img, title, cmap='gray', clim=None):
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
            if clim:
                im.set_clim(*clim)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        show(axes[0,0], c_img, "Clean MRI", clim=(0,1))
        show(axes[0,1], h2_img, "HM2 MRI", clim=(0,1))
        show(axes[0,2], gt_art, "GT Artifact", cmap='bwr', clim=(-0.2, 0.2))
        show(axes[0,3], p_art, "Pred Artifact", cmap='bwr', clim=(-0.2, 0.2))
        
        show(axes[1,0], p_img, "Stage-2 Recon", clim=(0,1))
        show(axes[1,1], np.abs(gt_art - p_art), "|GT - Pred| Art Error", cmap='hot', clim=(0, 0.1))
        show(axes[1,2], gt_art_grad, "GT Artifact Grad", cmap='magma', clim=(0, 0.2))
        show(axes[1,3], p_art_grad, "Pred Artifact Grad", cmap='magma', clim=(0, 0.2))
        
        show(axes[2,0], c_grad, "Clean Grad", cmap='magma', clim=(0, 0.2))
        show(axes[2,1], p_grad, "Recon Grad", cmap='magma', clim=(0, 0.2))
        
        c_fourier = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(c_img))))
        p_fourier = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(p_img))))
        
        show(axes[2,2], c_fourier, "Clean Fourier (Log)", cmap='plasma')
        show(axes[2,3], p_fourier, "Recon Fourier (Log)", cmap='plasma')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{subj_name}_slice_{sl}_audit.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    return subject_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="audit_outputs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Robust Model Loading
    stage1 = Stage1ArtifactPredictor(base_filters=Config.BASE_FILTERS_STAGE1).to(device)
    stage1_ckpt = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    stage1_state = stage1_ckpt.get("state_dict", stage1_ckpt)
    stage1_state = {k[7:] if k.startswith("module.") else k: v for k, v in stage1_state.items()}
    stage1.load_state_dict(stage1_state, strict=True)
    stage1.eval()
    
    stage2 = Stage2ArtifactGuidedReconstructor(
        base_filters=Config.BASE_FILTERS_STAGE2
    ).to(device)
    stage2_ckpt = torch.load(args.stage2_checkpoint, map_location=device, weights_only=False)
    stage2_state = stage2_ckpt.get("state_dict", stage2_ckpt)
    stage2_state = {k[7:] if k.startswith("module.") else k: v for k, v in stage2_state.items()}
    stage2.load_state_dict(stage2_state, strict=True)
    stage2.eval()
    
    sub_dirs = sorted(glob.glob(os.path.join(args.data_dir, "sub-*"))) + \
               sorted(glob.glob(os.path.join(args.data_dir, "subject_*")))
               
    if not sub_dirs:
        if os.path.exists(os.path.join(args.data_dir, "hm1.npy")):
            sub_dirs = [args.data_dir]
        else:
            raise FileNotFoundError(f"No valid test subjects found in {args.data_dir}")
            
    all_metrics = []
    for subj_path in sub_dirs:
        if os.path.exists(os.path.join(subj_path, "hm1.npy")):
            # Evaluate at 25%, 50%, 75%
            metrics = process_subject(subj_path, stage1, stage2, device, args.output_dir, slices=[64, 128, 192])
            all_metrics.extend(metrics)
            
    if not all_metrics:
        print("No valid subjects processed.")
        return
        
    print("\n" + "="*80)
    print("FINAL EVIDENCE REPORT (Averaged across all subjects and slices)")
    print("="*80)
    
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys() if k not in ["subj", "slice"]}
    
    print(f"HM2 Gradient Energy Ratio:   {avg['hm2_ge_ratio']:.4f}")
    print(f"Recon Gradient Energy Ratio: {avg['recon_ge_ratio']:.4f}\n")
    
    print(f"HM2 HF Retention (0.50):     {avg['hm2_hf_050']*100:.2f}%")
    print(f"Recon HF Retention (0.50):   {avg['recon_hf_050']*100:.2f}%")
    print(f"PSNR Improvement (Recon-HM2): +{avg['psnr_diff']:.2f} dB\n")
    
    print(f"Artifact Pearson Corr:       {avg['art_pearson']:.4f}")
    print(f"Excess Leakage:              {avg['excess_leakage']:.4f}\n")
    
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    
    # Logical Evaluation
    excess_leakage = avg['excess_leakage']
    hf_drop_recon = avg['hm2_hf_050'] - avg['recon_hf_050']
    ge_drop_recon = avg['hm2_ge_ratio'] - avg['recon_ge_ratio']
    psnr_imp = avg['psnr_diff']
    
    if avg['recon_ge_ratio'] < 0.20:
        print("D. Previous diagnostic metrics were flawed (but blur is real).")
        print("   Even with strict brain masking, GE ratio is extremely low. However, ")
    
    if excess_leakage > 0.15 and hf_drop_recon > 0.10:
        print("C. Both Stage-1 and Stage-2 significantly contribute.")
        print("   Stage-1 shows strong Excess Leakage, destroying anatomy, AND Stage-2 actively drops HF detail compared to HM2.")
    elif excess_leakage > 0.15:
        print("A. Stage-1 is the primary problem.")
        print("   Stage-1 shows massive Excess Leakage. It predicts anatomy instead of artifact.")
    elif hf_drop_recon > 0.10 and psnr_imp > 0:
        print("B. Stage-2 is the primary source of oversmoothing.")
        print("   Stage-2 drops HF detail drastically compared to HM2, yet PSNR improves. It acts as an objective-driven low-pass filter.")
    elif avg['recon_hf_050'] > 0.85:
        print("D. The apparent blur is primarily a visualization/display issue.")
        print("   HF Retention is very high. The mathematics do not show a blur.")
    else:
        print("E. Evidence is insufficient or mixed.")

if __name__ == "__main__":
    main()
