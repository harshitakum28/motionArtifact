import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from .trainer import BaseTrainer
from losses.charbonnier_loss import CharbonnierLoss
from losses.ssim_loss import SSIMLoss
from losses.focal_frequency_loss import FocalFrequencyLoss
from losses.gradient_loss import GradientLoss3D
from losses.preservation_loss import ArtifactAwarePreservationLoss

class Stage2Trainer(BaseTrainer):
    """
    Trainer for Stage 2: Artifact-Guided MRI Reconstruction.
    Assumes Stage 1 is pre-trained and frozen, or at least providing artifact maps.
    """
    def __init__(self, model_stage1, model_stage2, optimizer, train_loader, val_loader, config, device):
        super(Stage2Trainer, self).__init__(model_stage2, optimizer, train_loader, val_loader, config, device)
        self.model_stage1 = model_stage1
        
        # Explicitly freeze Stage 1
        self.model_stage1.eval()
        for param in self.model_stage1.parameters():
            param.requires_grad = False
            
        self.criterion_char = CharbonnierLoss(eps=1e-6).to(device)
        self.criterion_ssim = SSIMLoss(window_size=11, size_average=True).to(device)
        self.criterion_ffl = FocalFrequencyLoss(alpha=1.0).to(device)
        self.criterion_grad = GradientLoss3D().to(device)
        self.criterion_preserve = ArtifactAwarePreservationLoss(k=50.0).to(device)
        
        self.lambda_char = config.LAMBDA_CHAR
        self.lambda_ssim = config.LAMBDA_SSIM
        self.lambda_ffl = config.LAMBDA_FFL
        self.lambda_grad = config.LAMBDA_GRAD
        self.lambda_preserve = config.LAMBDA_PRESERVE

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        total_char, total_ssim, total_ffl, total_grad, total_preserve = 0.0, 0.0, 0.0, 0.0, 0.0
        total_char_w, total_ssim_w, total_ffl_w, total_grad_w, total_preserve_w = 0.0, 0.0, 0.0, 0.0, 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Stage 2 Train Epoch {self.current_epoch+1}")
        for batch in pbar:
            hm1 = batch['hm1'].to(self.device)
            hm2 = batch['hm2'].to(self.device)
            clean = batch['clean'].to(self.device)
            
            # 1. Get predicted artifact map from frozen Stage 1
            with torch.no_grad():
                pred_artifact = self.model_stage1(hm1, hm2)
            
            # 2. Train Stage 2
            self.optimizer.zero_grad()
            
            # Pure residual reconstruction is enforced by the model: pred_clean, raw_correction
            pred_clean, raw_correction = self.model(hm2, pred_artifact, return_correction=True)
            
            # Calculate raw, unweighted losses on the raw output without clamping
            l_char = self.criterion_char(pred_clean, clean)
            l_ssim = self.criterion_ssim(pred_clean, clean)
            l_ffl = self.criterion_ffl(pred_clean, clean)
            l_grad = self.criterion_grad(pred_clean, clean)
            l_preserve = self.criterion_preserve(raw_correction, hm2, clean)
            
            # Weighted losses
            w_char = self.lambda_char * l_char
            w_ssim = self.lambda_ssim * l_ssim
            w_ffl = self.lambda_ffl * l_ffl
            w_grad = self.lambda_grad * l_grad
            w_preserve = self.lambda_preserve * l_preserve
            
            loss = w_char + w_ssim + w_ffl + w_grad + w_preserve
            loss.backward()
            self.optimizer.step()
            
            bs = hm2.size(0)
            total_loss += loss.item() * bs
            
            total_char += l_char.item() * bs
            total_ssim += l_ssim.item() * bs
            total_ffl += l_ffl.item() * bs
            total_grad += l_grad.item() * bs
            total_preserve += l_preserve.item() * bs
            
            total_char_w += w_char.item() * bs
            total_ssim_w += w_ssim.item() * bs
            total_ffl_w += w_ffl.item() * bs
            total_grad_w += w_grad.item() * bs
            total_preserve_w += w_preserve.item() * bs
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'WGrad': f"{w_grad.item():.4f}",
                'WPres': f"{w_preserve.item():.4f}"
            })
            
        n = len(self.train_loader.dataset)
        self.logger.info(f"Train Epoch {self.current_epoch+1} -> Total Loss: {total_loss/n:.4f}")
        self.logger.info(f"  Raw: Char={total_char/n:.5f}, SSIM={total_ssim/n:.5f}, FFL={total_ffl/n:.5f}, Grad={total_grad/n:.5f}, Pres={total_preserve/n:.5f}")
        self.logger.info(f"  Wtd: WChar={total_char_w/n:.5f}, WSSIM={total_ssim_w/n:.5f}, WFFL={total_ffl_w/n:.5f}, WGrad={total_grad_w/n:.5f}, WPres={total_preserve_w/n:.5f}")
        return total_loss / n

    def validate_epoch(self):
        if len(self.val_loader.dataset) == 0:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        total_char, total_ssim, total_ffl, total_grad, total_preserve = 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Metric tracking is done on clamped outputs
        self.metrics.reset()
        saved_visualization = False
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc=f"Stage 2 Val Epoch {self.current_epoch+1}")):
                hm1 = batch['hm1'].to(self.device)
                hm2 = batch['hm2'].to(self.device)
                clean = batch['clean'].to(self.device)
                
                pred_artifact = self.model_stage1(hm1, hm2)
                pred_clean, raw_correction = self.model(hm2, pred_artifact, return_correction=True)
                
                l_char = self.criterion_char(pred_clean, clean)
                l_ssim = self.criterion_ssim(pred_clean, clean)
                l_ffl = self.criterion_ffl(pred_clean, clean)
                l_grad = self.criterion_grad(pred_clean, clean)
                l_preserve = self.criterion_preserve(raw_correction, hm2, clean)
                
                w_char = self.lambda_char * l_char
                w_ssim = self.lambda_ssim * l_ssim
                w_ffl = self.lambda_ffl * l_ffl
                w_grad = self.lambda_grad * l_grad
                w_preserve = self.lambda_preserve * l_preserve
                
                loss = w_char + w_ssim + w_ffl + w_grad + w_preserve
                
                bs = hm2.size(0)
                total_loss += loss.item() * bs
                total_char += l_char.item() * bs
                total_ssim += l_ssim.item() * bs
                total_ffl += l_ffl.item() * bs
                total_grad += l_grad.item() * bs
                total_preserve += l_preserve.item() * bs
                
                # Compute Evaluation Metrics using clamped output
                clamped_pred = pred_clean.clamp(0, 1)
                self.metrics.update(clamped_pred, clean)
                
                # Visual Diagnostics for the first batch
                if not saved_visualization:
                    saved_visualization = True
                    gt_correction = clean - hm2
                    self._save_validation_visuals(hm2, clean, clamped_pred, gt_correction, raw_correction)
                
        n = len(self.val_loader.dataset)
        self.logger.info(f"Val Epoch {self.current_epoch+1} -> Total Loss: {total_loss/n:.4f}")
        self.logger.info(f"  Raw: Char={total_char/n:.5f}, SSIM={total_ssim/n:.5f}, FFL={total_ffl/n:.5f}, Grad={total_grad/n:.5f}, Pres={total_preserve/n:.5f}")
        return total_loss / n

    def _save_validation_visuals(self, hm2, clean, pred_clean, gt_correction, raw_correction):
        try:
            out_dir = os.path.join(self.config.CHECKPOINT_DIR, "visuals")
            os.makedirs(out_dir, exist_ok=True)
            
            # Find the most informative slice in the batch (avoid empty edge slices)
            best_b, best_d = 0, hm2.shape[2] // 2
            max_mean = -1
            
            for b in range(hm2.shape[0]):
                for d in range(hm2.shape[2]):
                    slice_mean = clean[b, 0, d].mean().item()
                    if slice_mean > max_mean:
                        max_mean = slice_mean
                        best_b, best_d = b, d
                        
            b_idx, d_idx = best_b, best_d
            
            hm2_img = hm2[b_idx, 0, d_idx].cpu().numpy()
            clean_img = clean[b_idx, 0, d_idx].cpu().numpy()
            pred_img = pred_clean[b_idx, 0, d_idx].cpu().numpy()
            gt_corr_img = gt_correction[b_idx, 0, d_idx].cpu().numpy()
            pred_corr_img = raw_correction[b_idx, 0, d_idx].cpu().numpy()
            
            abs_corr_err = torch.abs(gt_correction - raw_correction)[b_idx, 0, d_idx].cpu().numpy()
            abs_recon_err = torch.abs(clean - pred_clean)[b_idx, 0, d_idx].cpu().numpy()
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            axes[0,0].imshow(hm2_img, cmap='gray', vmin=0, vmax=1)
            axes[0,0].set_title("HM2")
            axes[0,1].imshow(clean_img, cmap='gray', vmin=0, vmax=1)
            axes[0,1].set_title("Clean")
            axes[0,2].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
            axes[0,2].set_title("Final Reconstruction")
            axes[0,3].imshow(abs_recon_err, cmap='hot', vmin=0, vmax=0.1)
            axes[0,3].set_title("Abs Recon Error")
            
            max_c = max(abs(gt_corr_img).max(), abs(pred_corr_img).max()) * 0.8 # Dynamic visual clip
            if max_c == 0: max_c = 0.1
            
            axes[1,0].imshow(gt_corr_img, cmap='bwr', vmin=-max_c, vmax=max_c)
            axes[1,0].set_title("GT Correction")
            axes[1,1].imshow(pred_corr_img, cmap='bwr', vmin=-max_c, vmax=max_c)
            axes[1,1].set_title("Predicted Correction")
            axes[1,2].imshow(abs_corr_err, cmap='hot', vmin=0, vmax=max_c)
            axes[1,2].set_title("Abs Correction Error")
            axes[1,3].axis('off')
            
            for ax in axes.flatten():
                ax.axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"epoch_{self.current_epoch+1}.png"))
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to save visual diagnostics: {e}")
