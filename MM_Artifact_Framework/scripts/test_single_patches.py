import os
import sys
import torch
import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
from torchvision import transforms

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from models.stage1 import Stage1ArtifactPredictor
from models.stage2 import Stage2ArtifactGuidedReconstructor
from inference.inference import MRIArtifactFrameworkInference
from utils.checkpoint import load_checkpoint_strict_for_inference
from datasets.transforms import ToTensor

def main():
    parser = argparse.ArgumentParser(description="Test Specific Patches")
    parser.add_argument("--stage1_checkpoint", type=str, required=True, help="Path to Stage 1 checkpoint")
    parser.add_argument("--stage2_checkpoint", type=str, required=True, help="Path to Stage 2 checkpoint")
    parser.add_argument("--data_dir", type=str, default="", help="Directory containing test data. Uses Config default if empty.")
    parser.add_argument("--patches", type=int, nargs='+', default=[8], help="List of patch indices to test (0-15)")
    args = parser.parse_args()

    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # DIRECTORY AUDIT & SUBJECT DISCOVERY
    # ---------------------------------------------------------
    if args.data_dir:
        resolved_data_dir = args.data_dir
    elif hasattr(config, 'TEST_DATA_DIR') and config.TEST_DATA_DIR:
        resolved_data_dir = config.TEST_DATA_DIR
    else:
        resolved_data_dir = config.DATA_DIR

    sub_dirs = sorted(glob.glob(os.path.join(resolved_data_dir, "sub-*"))) + \
               sorted(glob.glob(os.path.join(resolved_data_dir, "subject_*"))) + \
               [os.path.join(resolved_data_dir, d) for d in os.listdir(resolved_data_dir) if os.path.isdir(os.path.join(resolved_data_dir, d)) and os.path.exists(os.path.join(resolved_data_dir, d, "hm1.npy"))]
               
    # Remove duplicates
    sub_dirs = list(set(sub_dirs))

    if not sub_dirs:
        print(f"ERROR: No subjects found in {resolved_data_dir} containing hm1.npy")
        sys.exit(1)
        
    print(f"Discovered {len(sub_dirs)} subjects.")

    # ---------------------------------------------------------
    # LOAD MODELS
    # ---------------------------------------------------------
    print("Loading models...")
    stage1 = Stage1ArtifactPredictor(base_filters=config.BASE_FILTERS_STAGE1).to(device)
    load_checkpoint_strict_for_inference(args.stage1_checkpoint, stage1, device)
    stage1.eval()
    
    stage2 = Stage2ArtifactGuidedReconstructor(
        base_filters=config.BASE_FILTERS_STAGE2,
        num_residual_blocks=config.NUM_RESIDUAL_BLOCKS
    ).to(device)
    load_checkpoint_strict_for_inference(args.stage2_checkpoint, stage2, device)
    stage2.eval()
    
    inference_engine = MRIArtifactFrameworkInference(stage1, stage2, device)
    transform = transforms.Compose([ToTensor()])
    
    ckpt_name = os.path.splitext(os.path.basename(args.stage2_checkpoint))[0]
    out_base_dir = os.path.join(PROJECT_ROOT, "output_patches", ckpt_name)
    os.makedirs(out_base_dir, exist_ok=True)

    # ---------------------------------------------------------
    # PROCESS PATCHES
    # ---------------------------------------------------------
    for subj_path in sub_dirs:
        subj_name = os.path.basename(subj_path)
        print(f"\nProcessing {subj_name}...")
        
        # Load full volumes (Pre-normalized as per your request)
        hm1_vol = np.load(os.path.join(subj_path, "hm1.npy"))
        hm2_vol = np.load(os.path.join(subj_path, "hm2.npy"))
        clean_vol = np.load(os.path.join(subj_path, "clean.npy"))
        
        for patch_idx in args.patches:
            if patch_idx < 0 or patch_idx > 15:
                print(f"Skipping invalid patch index: {patch_idx}")
                continue
                
            print(f"  Testing Patch {patch_idx}...")
            
            # Extract specific patch
            start_z = patch_idx * 16
            end_z = start_z + 16
            
            p_hm1 = hm1_vol[:, :, start_z:end_z]
            p_hm2 = hm2_vol[:, :, start_z:end_z]
            p_clean = clean_vol[:, :, start_z:end_z]
            
            # Convert to tensors
            sample_patch = transform({'hm1': p_hm1, 'hm2': p_hm2})
            hm1_pt = sample_patch['hm1'].unsqueeze(0).to(device)
            hm2_pt = sample_patch['hm2'].unsqueeze(0).to(device)
            
            # Run Inference
            with torch.no_grad():
                pred_artifact = inference_engine.infer_stage1_patch(hm1_pt, hm2_pt)
                pred_clean_pt = inference_engine.infer_stage2_patch(hm2_pt, pred_artifact)
                
            # Convert back to numpy (clip to [0, 1] for visualization)
            pred_clean_np = pred_clean_pt.squeeze().cpu().numpy().transpose((1, 2, 0))
            pred_clean_np = np.clip(pred_clean_np, 0.0, 1.0)
            
            # ---------------------------------------------------------
            # VISUALIZE (Separate exact size images)
            # ---------------------------------------------------------
            vis_dir = os.path.join(out_base_dir, subj_name)
            os.makedirs(vis_dir, exist_ok=True)
            
            # If the patch contains global slice 128, show that specific slice to match test.py.
            # Otherwise, default to the middle slice of the patch (8)
            if start_z <= 128 < end_z:
                slice_idx = 128 - start_z
                print(f"    -> Visualizing Global Slice 128 (Local slice index: {slice_idx})")
            else:
                slice_idx = 8
                print(f"    -> Visualizing Middle Slice (Local slice index: {slice_idx})")
            
            out_hm1 = os.path.join(vis_dir, f"patch_{patch_idx}_slice_{slice_idx}_hm1.png")
            out_hm2 = os.path.join(vis_dir, f"patch_{patch_idx}_slice_{slice_idx}_hm2.png")
            out_clean = os.path.join(vis_dir, f"patch_{patch_idx}_slice_{slice_idx}_clean.png")
            out_pred = os.path.join(vis_dir, f"patch_{patch_idx}_slice_{slice_idx}_pred.png")
            
            # plt.imsave saves the numpy array exactly pixel-for-pixel without borders or axes
            plt.imsave(out_hm1, p_hm1[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
            plt.imsave(out_hm2, p_hm2[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
            plt.imsave(out_clean, p_clean[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
            plt.imsave(out_pred, pred_clean_np[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
            
            print(f"  -> Saved separate exact-size images to {vis_dir}")

    print("\nPatch testing complete!")

if __name__ == '__main__':
    main()
