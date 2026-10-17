import torch
from tqdm import tqdm
from .trainer import BaseTrainer
from losses.artifact_loss import ArtifactLoss

class Stage1Trainer(BaseTrainer):
    """
    Trainer for Stage 1: Artifact Predictor.
    """
    def __init__(self, model, optimizer, train_loader, val_loader, config, device):
        super(Stage1Trainer, self).__init__(model, optimizer, train_loader, val_loader, config, device)
        self.criterion = ArtifactLoss().to(device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(self.train_loader, desc=f"Stage 1 Train Epoch {self.current_epoch+1}"):
            hm1 = batch['hm1'].to(self.device)
            hm2 = batch['hm2'].to(self.device)
            clean = batch['clean'].to(self.device)
            
            self.optimizer.zero_grad()
            
            pred_artifact = self.model(hm1, hm2)
            loss = self.criterion(pred_artifact, hm2, clean)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * hm1.size(0)
            
        return total_loss / len(self.train_loader.dataset)

    def validate_epoch(self):
        if len(self.val_loader.dataset) == 0:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        
        saved_visualization = False
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Stage 1 Val Epoch {self.current_epoch+1}"):
                hm1 = batch['hm1'].to(self.device)
                hm2 = batch['hm2'].to(self.device)
                clean = batch['clean'].to(self.device)
                
                pred_artifact = self.model(hm1, hm2)
                loss = self.criterion(pred_artifact, hm2, clean)
                
                total_loss += loss.item() * hm1.size(0)
                
                if not saved_visualization:
                    saved_visualization = True
                    gt_artifact = hm2 - clean
                    self._save_validation_visuals(hm1, hm2, clean, gt_artifact, pred_artifact)
                
        # Metrics aren't as relevant for just the artifact map, 
        # but we must return the validation loss for checkpointing.
        return total_loss / len(self.val_loader.dataset)

    def _save_validation_visuals(self, hm1, hm2, clean, gt_artifact, pred_artifact):
        try:
            import os
            import matplotlib.pyplot as plt
            out_dir = os.path.join(self.config.CHECKPOINT_DIR, "visuals")
            os.makedirs(out_dir, exist_ok=True)
            
            best_b, best_d = 0, hm2.shape[2] // 2
            max_mean = -1
            
            for b in range(hm2.shape[0]):
                for d in range(hm2.shape[2]):
                    slice_mean = clean[b, 0, d].mean().item()
                    if slice_mean > max_mean:
                        max_mean = slice_mean
                        best_b, best_d = b, d
                        
            b_idx, d_idx = best_b, best_d
            
            hm1_img = hm1[b_idx, 0, d_idx].cpu().numpy()
            hm2_img = hm2[b_idx, 0, d_idx].cpu().numpy()
            clean_img = clean[b_idx, 0, d_idx].cpu().numpy()
            gt_art_img = gt_artifact[b_idx, 0, d_idx].cpu().numpy()
            pred_art_img = pred_artifact[b_idx, 0, d_idx].cpu().numpy()
            
            err_img = torch.abs(gt_artifact - pred_artifact)[b_idx, 0, d_idx].cpu().numpy()
            
            fig, axes = plt.subplots(1, 6, figsize=(30, 5))
            axes[0].imshow(hm1_img, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title("HM1")
            axes[1].imshow(hm2_img, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title("HM2")
            axes[2].imshow(clean_img, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title("Clean")
            
            max_c = max(abs(gt_art_img).max(), abs(pred_art_img).max()) * 0.8
            if max_c == 0: max_c = 0.1
            
            axes[3].imshow(gt_art_img, cmap='bwr', vmin=-max_c, vmax=max_c)
            axes[3].set_title("GT Artifact")
            axes[4].imshow(pred_art_img, cmap='bwr', vmin=-max_c, vmax=max_c)
            axes[4].set_title("Predicted Artifact")
            axes[5].imshow(err_img, cmap='hot', vmin=0, vmax=max_c)
            axes[5].set_title("Abs Error")
            
            for ax in axes.flatten():
                ax.axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"epoch_{self.current_epoch+1}.png"))
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to save visual diagnostics: {e}")
