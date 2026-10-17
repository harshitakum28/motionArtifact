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
from datasets.transforms import ToTensor
from torchvision import transforms

# --- Utility Math Functions ---
def compute_gradient_magnitude(img):
    """Computes Sobel gradient magnitude of a 2D image."""
    gx = ndimage.sobel(img, axis=0)
    gy = ndimage.sobel(img, axis=1)
    return np.sqrt(gx**2 + gy**2)

def compute_fourier_spectrum(img):
    """Computes the log-magnitude Fourier spectrum."""
    F = np.fft.fftshift(np.fft.fft2(img))
    return np.log(1 + np.abs(F))

def compute_hf_energy(img, threshold_ratio=0.5):
    """Computes high-frequency energy using a radial mask."""
    F = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(F)**2
    
    H, W = img.shape
    Y, X = np.ogrid[-H/2:H/2, -W/2:W/2]
    R = np.sqrt(X**2 + Y**2)
    R_max = min(H, W) / 2.0
    
    hf_mask = R > (threshold_ratio * R_max)
    hf_energy = np.sum(power[hf_mask])
    return hf_energy

def compute_pearson_corr(img1, img2):
    """Computes Pearson correlation coefficient between two 2D images."""
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]

def process_subject(subj_path, stage1, stage2, device, output_dir, slice_idx=128):
    subj_name = os.path.basename(subj_path.rstrip("/\\"))
    print(f"\nProcessing {subj_name}...")
    
    # Load normalized volumes
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
        
    # Clamp clean to [0,1]
    pred_clean_vol = np.clip(pred_clean_vol, 0.0, 1.0)
    
    # Extract central slice
    sl = slice_idx
    c_img = clean_vol[:, :, sl]
    h2_img = hm2_vol[:, :, sl]
    p_img = pred_clean_vol[:, :, sl]
    
    gt_art = h2_img - c_img
    p_art_img = pred_artifact_vol[:, :, sl]
    
    # Computations
    c_grad = compute_gradient_magnitude(c_img)
    h2_grad = compute_gradient_magnitude(h2_img)
    p_grad = compute_gradient_magnitude(p_img)
    
    c_ge = np.sum(c_grad**2)
    p_ge = np.sum(p_grad**2)
    ge_ratio = p_ge / (c_ge + 1e-8)
    
    c_hf = compute_hf_energy(c_img)
    p_hf = compute_hf_energy(p_img)
    hf_ratio = p_hf / (c_hf + 1e-8)
    
    # Artifact Metrics
    art_mae = np.mean(np.abs(gt_art - p_art_img))
    art_mse = np.mean((gt_art - p_art_img)**2)
    art_pearson = compute_pearson_corr(gt_art, p_art_img)
    
    gt_art_grad = compute_gradient_magnitude(gt_art)
    p_art_grad = compute_gradient_magnitude(p_art_img)
    art_grad_corr = compute_pearson_corr(gt_art_grad, p_art_grad)
    
    leakage_corr = compute_pearson_corr(p_art_grad, c_grad)
    
    metrics = {
        "subj": subj_name,
        "ge_ratio": ge_ratio,
        "hf_retention": hf_ratio * 100,
        "art_mae": art_mae,
        "art_mse": art_mse,
        "art_pearson": art_pearson,
        "art_grad_corr": art_grad_corr,
        "leakage_corr": leakage_corr
    }
    
    # Plotting
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"Diagnostic Analysis - {subj_name} (Slice {sl})", fontsize=16)
    
    def show(ax, img, title, cmap='gray', clim=None):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        if clim:
            im.set_clim(*clim)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show(axes[0,0], c_img, "1. Clean MRI", clim=(0,1))
    show(axes[0,1], h2_img, "2. HM2 MRI", clim=(0,1))
    show(axes[0,2], gt_art, "3. GT Artifact (HM2-Clean)", cmap='bwr', clim=(-0.2, 0.2))
    show(axes[0,3], p_art_img, "4. Stage-1 Predicted Artifact", cmap='bwr', clim=(-0.2, 0.2))
    
    show(axes[1,0], p_img, "5. Stage-2 Reconstructed", clim=(0,1))
    err = np.abs(c_img - p_img)
    show(axes[1,1], err, "6. Absolute Error", cmap='hot', clim=(0, 0.1))
    show(axes[1,2], gt_art_grad, "7. GT Artifact Gradient", cmap='magma')
    show(axes[1,3], p_art_grad, "8. Pred Artifact Gradient", cmap='magma')
    
    show(axes[2,0], c_grad, "9. Clean Gradient", cmap='magma')
    show(axes[2,1], p_grad, "10. Recon Gradient", cmap='magma')
    
    c_fourier = compute_fourier_spectrum(c_img)
    p_fourier = compute_fourier_spectrum(p_img)
    
    show(axes[2,2], c_fourier, "11. Clean Fourier (Log-Mag)", cmap='plasma')
    show(axes[2,3], p_fourier, "12. Recon Fourier (Log-Mag)", cmap='plasma')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{subj_name}_diagnostic.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Diagnose Oversmoothing and Blur.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="diagnostic_outputs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Models
    stage1 = Stage1ArtifactPredictor(
        base_filters=Config.BASE_FILTERS_STAGE1
    ).to(device)

    stage1_ckpt = torch.load(
        args.stage1_checkpoint,
        map_location=device,
        weights_only=False
    )

    stage1_state = stage1_ckpt.get("state_dict", stage1_ckpt)

    # Remove DataParallel "module." prefix if present
    stage1_state = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in stage1_state.items()
    }

    stage1.load_state_dict(stage1_state, strict=True)
    stage1.eval()

    print(
        f"Loaded Stage1 checkpoint: {args.stage1_checkpoint}\n"
        f"Epoch: {stage1_ckpt.get('epoch', 'N/A')} | "
        f"Best Loss: {stage1_ckpt.get('best_loss', 'N/A')}"
    )


    stage2 = Stage2ArtifactGuidedReconstructor(
        base_filters=Config.BASE_FILTERS_STAGE2
    ).to(device)

    stage2_ckpt = torch.load(
        args.stage2_checkpoint,
        map_location=device,
        weights_only=False
    )

    stage2_state = stage2_ckpt.get("state_dict", stage2_ckpt)

    # Remove DataParallel "module." prefix if present
    stage2_state = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in stage2_state.items()
    }

    stage2.load_state_dict(stage2_state, strict=True)
    stage2.eval()

    print(
        f"Loaded Stage2 checkpoint: {args.stage2_checkpoint}\n"
        f"Epoch: {stage2_ckpt.get('epoch', 'N/A')} | "
        f"Best Loss: {stage2_ckpt.get('best_loss', 'N/A')}"
    )
    
    # Find Subjects
    sub_dirs = sorted(glob.glob(os.path.join(args.data_dir, "sub-*"))) + \
               sorted(glob.glob(os.path.join(args.data_dir, "subject_*")))
               
    if not sub_dirs:
        if os.path.exists(os.path.join(args.data_dir, "hm1.npy")):
            sub_dirs = [args.data_dir]
        else:
            raise FileNotFoundError(f"No valid test subjects found in {args.data_dir}")
            
    print(f"Found {len(sub_dirs)} subjects for diagnosis.")
    
    all_metrics = []
    for subj_path in sub_dirs:
        if os.path.exists(os.path.join(subj_path, "hm1.npy")):
            metrics = process_subject(subj_path, stage1, stage2, device, args.output_dir, slice_idx=128)
            all_metrics.append(metrics)
            
    if not all_metrics:
        print("No valid subjects processed.")
        return
        
    # Averages
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys() if k != "subj"}
    
    print("\n" + "="*60)
    print("DIAGNOSTIC REPORT")
    print("="*60)
    print(f"{'Subject':<15} | {'GE Ratio':<10} | {'HF Ret %':<10} | {'Art Pearson':<12} | {'Leakage Corr':<12}")
    print("-" * 60)
    for m in all_metrics:
        print(f"{m['subj']:<15} | {m['ge_ratio']:.4f}     | {m['hf_retention']:.2f}%     | {m['art_pearson']:.4f}       | {m['leakage_corr']:.4f}")
    print("-" * 60)
    print(f"{'AVERAGE':<15} | {avg_metrics['ge_ratio']:.4f}     | {avg_metrics['hf_retention']:.2f}%     | {avg_metrics['art_pearson']:.4f}       | {avg_metrics['leakage_corr']:.4f}")
    
    # Conclusion Logic
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    leakage = avg_metrics['leakage_corr']
    hf_ret = avg_metrics['hf_retention']
    
    if leakage > 0.4 and hf_ret < 75.0:
        conclusion = "C. Both Stage-1 and Stage-2 contribute to blur."
        reason = (f"Stage-1 is heavily leaking anatomical edges into the artifact map (Leakage Corr = {leakage:.4f} > 0.40).\n"
                  f"Simultaneously, Stage-2 is failing to preserve high frequencies (HF Retention = {hf_ret:.2f}% < 75%).")
    elif leakage > 0.4:
        conclusion = "A. Stage-1 artifact guidance is damaging anatomy."
        reason = (f"Stage-1 has strong structural leakage (Leakage Corr = {leakage:.4f} > 0.40), "
                  "meaning it mistakenly identifies clean anatomical edges as artifacts and guides Stage-2 to remove them.")
    elif hf_ret < 75.0:
        conclusion = "B. Stage-2 itself is oversmoothing."
        reason = (f"Stage-1 structural leakage is low (Leakage Corr = {leakage:.4f} < 0.40), but "
                  f"Stage-2 fails to retain high-frequency details (HF Retention = {hf_ret:.2f}% < 75%). "
                  "This indicates the reconstruction network (Stage 2) is acting as a low-pass filter.")
    else:
        conclusion = "D. The apparent blur is primarily a visualization/display issue."
        reason = (f"Structural leakage is low ({leakage:.4f}) and High-Frequency retention is high ({hf_ret:.2f}%). "
                  "The underlying structural integrity and high frequencies are preserved.")
                  
    print(conclusion)
    print("\nSupporting Evidence:")
    print(reason)
    print("="*60)

if __name__ == "__main__":
    main()
