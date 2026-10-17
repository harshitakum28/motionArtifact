import os
import torch
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint
from utils.metrics import EvaluationMetrics

class BaseTrainer:
    """
    Base trainer class containing generic training loops.
    """
    def __init__(self, model, optimizer, train_loader, val_loader, config, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.logger = get_logger(self.__class__.__name__, self.config.LOG_DIR)
        self.metrics = EvaluationMetrics()
        
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.window_best_loss = float('inf')
        self.best_psnr = -float('inf')
        self.best_ssim = -float('inf')

    def train_epoch(self):
        raise NotImplementedError

    def validate_epoch(self):
        raise NotImplementedError

    def train(self, resume_path=None):
        """
        Main training loop.
        """
        if resume_path and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            # Don't load optimizer if fine-tuning (handled in train.py logic, but for standard resume we do it here)
            if 'optimizer' in checkpoint and hasattr(self, 'optimizer') and self.optimizer:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                except Exception as e:
                    self.logger.warning(f"Could not load optimizer state: {e}")
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.best_psnr = checkpoint.get('best_psnr', -float('inf'))
            self.best_ssim = checkpoint.get('best_ssim', -float('inf'))
            self.logger.info(f"Resumed training from {resume_path} at epoch {self.current_epoch+1}")

        for epoch in range(self.current_epoch, self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Metrics are computed during validate_epoch
            val_metrics = self.metrics.get_metrics()
            val_psnr = val_metrics.get('PSNR', 0.0)
            val_ssim = val_metrics.get('SSIM', 0.0)
            
            self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}] "
                             f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                             f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f} | MSE: {val_metrics.get('MSE', 0.0):.6f}")
            
            # Checkpoint Tracking
            is_best_loss = val_loss < self.best_loss
            
            if is_best_loss:
                self.best_loss = val_loss
                
            # Window best tracking for Stage 2
            is_window_best = False
            window_best_name = None
            if getattr(self.config, 'STAGE', None) == 2:
                if epoch % 5 == 0:
                    self.window_best_loss = float('inf')
                    
                if val_loss < self.window_best_loss:
                    self.window_best_loss = val_loss
                    is_window_best = True
                    window_start = (epoch // 5) * 5 + 1
                    window_end = window_start + 4
                    window_best_name = f"epoch{window_start}-{window_end}_best_loss.pth"

            model_state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                
            state = {
                'epoch': epoch + 1,
                'state_dict': model_state_dict,
                'best_loss': self.best_loss,
                'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            }
            
            save_kwargs = {
                'config': self.config
            }
            
            if getattr(self.config, 'STAGE', None) == 2:
                if (epoch + 1) % 5 == 0:
                    save_kwargs['interval_epoch'] = epoch + 1
                if is_window_best:
                    save_kwargs['is_window_best'] = True
                    save_kwargs['window_best_name'] = window_best_name
                
            save_checkpoint(
                state, 
                is_best_loss, 
                self.config.CHECKPOINT_DIR, 
                **save_kwargs
            )
            
        return os.path.join(self.config.CHECKPOINT_DIR, 'model_best_loss.pth')
