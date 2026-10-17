import torch
import torch.nn.functional as F
import math

def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculates Mean Squared Error."""
    return F.mse_loss(pred, target).item()

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted tensor.
        target: Target tensor.
        data_range: Maximum possible pixel value of the image.
    """
    mse = calculate_mse(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * math.log10((data_range ** 2) / mse)

def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Calculates Structural Similarity Index (SSIM).
    We can reuse the 3D SSIM from our losses, but for metrics, we often compute 
    it independently or on the CPU using scikit-image if preferred.
    To avoid dependencies, we implement a basic PyTorch 2D/3D evaluation wrapper.
    """
    # For a robust metric, we reuse the SSIM logic from losses but detached
    from losses.ssim_loss import ssim_3d, create_window_3d
    
    with torch.no_grad():
        window_size = 11
        channel = pred.size(1)
        window = create_window_3d(window_size, channel).to(pred.device).type_as(pred)
        
        # Scaling constants based on data_range if needed, but ssim_3d uses defaults C1, C2 
        # which assume data in [0, 1].
        val = ssim_3d(pred, target, window_size, window, size_average=True)
        return val.item()

class EvaluationMetrics:
    """Helper class to compute and store all metrics."""
    def __init__(self, data_range=1.0):
        self.data_range = data_range
        self.reset()
        
    def reset(self):
        self.mse_sum = 0.0
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.count = 0
        
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.mse_sum += calculate_mse(pred, target)
        self.psnr_sum += calculate_psnr(pred, target, self.data_range)
        self.ssim_sum += calculate_ssim(pred, target, self.data_range)
        self.count += 1
        
    def get_metrics(self):
        if self.count == 0:
            return {"MSE": 0.0, "PSNR": 0.0, "SSIM": 0.0}
        
        return {
            "MSE": self.mse_sum / self.count,
            "PSNR": self.psnr_sum / self.count,
            "SSIM": self.ssim_sum / self.count
        }
