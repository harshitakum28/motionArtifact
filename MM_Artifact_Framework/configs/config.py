import os

class Config:
    """
    Centralized configuration for the MRI Motion Artifact Removal Framework.
    """
    # Project Paths
    if os.path.exists('/kaggle'):
        ROOT_DIR = "/kaggle/working"
        DATA_DIR = "/kaggle/input/datasets/premkumarsingh15029/dataset/kaggle dataset/data"
        TEST_DATA_DIR = "/kaggle/input/datasets/premkumarsingh15029/dataset/kaggle dataset/data"
        CHECKPOINT_DIR = "/kaggle/working/result"
    else:
        ROOT_DIR = r"C:\Users\bitpa\OneDrive\Desktop\cordinate attention module\MRI_Artifact_Framework"
        DATA_DIR = r"C:\Users\bitpa\OneDrive\Desktop\kaggle dataset\data"
        TEST_DATA_DIR = r"C:\Users\bitpa\OneDrive\Desktop\kaggle dataset\data"
        CHECKPOINT_DIR = r"C:\Users\bitpa\OneDrive\Desktop\result"
    LOG_DIR = os.path.join(ROOT_DIR, "logs")

    # Google Drive Backup
    BACKUP_TO_GDRIVE = True
    GDRIVE_FOLDER_ID = "1834nl9whsolK_sfSM2Av_AcWVCIYkYFh"

    # Training Parameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-6
    NUM_WORKERS = 4
    
    # Loss Weights (Stage-2)
    LAMBDA_CHAR = 1.0    # Charbonnier Loss
    LAMBDA_SSIM = 0.5    # SSIM Loss
    LAMBDA_FFL = 0.1     # Focal Frequency Loss
    LAMBDA_GRAD = 1.0    # 3D Spatial Gradient Loss
    LAMBDA_PRESERVE = 2.0 # Artifact-Aware Preservation Loss
    
    # Stage 1 Loss Weight
    ALPHA = 1.0      # Stage 1 Artifact Loss (if end-to-end)

    # Model Hyperparameters
    BASE_FILTERS_STAGE1 = 16
    BASE_FILTERS_STAGE2 = 32
    NUM_RESIDUAL_BLOCKS = 4
    COORD_ATT_REDUCTION = 16
    
    # Misc
    SEED = 42
    DEVICE = "cuda" # or "cpu"

    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
