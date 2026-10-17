import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and PyTorch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some cudnn operations might be non-deterministic.
    # Setting these flags ensures reproducibility at the cost of some performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
