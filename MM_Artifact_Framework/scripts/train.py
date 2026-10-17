import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

# Add parent directory to path to import modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from datasets.dataset import MRIMotionArtifactDataset
from models.stage1 import Stage1ArtifactPredictor
from models.stage2 import Stage2ArtifactGuidedReconstructor
from trainers.train_stage1 import Stage1Trainer
from trainers.train_stage2 import Stage2Trainer
from utils.checkpoint import load_state_dict_flexible
import glob
import random

def main():
    parser = argparse.ArgumentParser(description="Train MRI Artifact Removal Framework")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True, help="Training stage (1 or 2, 3 for both)")
    parser.add_argument("--stage1_checkpoint", type=str, default="", help="Path to pre-trained Stage 1 checkpoint (needed for Stage 2)")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument("--finetune_checkpoint", type=str, default="", help="Path to Stage 2 checkpoint to fine-tune from (loads weights only, starts fresh optimizer)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override total number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU training using torch.nn.DataParallel")
    parser.add_argument("--experiment_name", type=str, default="stage2_final_detail_preserved", help="Directory name for checkpoints")
    
    # New debug arguments
    parser.add_argument("--debug_one_batch", action="store_true", help="Print shapes, ranges, losses heavily for the first batch")
    parser.add_argument("--sanity_overfit", action="store_true", help="Train on a small subset (1-2 subjects) to test overfitting")
    args = parser.parse_args()

    # Setup configurations
    config = Config()
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
        
    base_checkpoint_dir = config.CHECKPOINT_DIR
    
    if args.stage == 1:
        config.CHECKPOINT_DIR = os.path.join(base_checkpoint_dir, "stage1")
    elif args.stage == 2:
        config.CHECKPOINT_DIR = os.path.join(base_checkpoint_dir, args.experiment_name)
    elif args.stage == 3:
        pass
        
    config.setup_dirs()
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if config.DEVICE == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Training will fall back to CPU. Please turn on GPU in Kaggle Notebook settings (Accelerator -> GPU).")
    
    # Dataloaders
    train_dir = os.path.join(config.DATA_DIR, "train")
    val_dir = os.path.join(config.DATA_DIR, "val")
    
    if os.path.exists(train_dir) and os.path.exists(val_dir) and not args.sanity_overfit:
        train_dataset = MRIMotionArtifactDataset(data_dir=train_dir)
        val_dataset = MRIMotionArtifactDataset(data_dir=val_dir)
    else:
        print(f"No train/val folders found in {config.DATA_DIR} or sanity_overfit enabled.")
        # Find all valid subjects
        subject_dirs = sorted(glob.glob(os.path.join(config.DATA_DIR, "sub-*")))
        if not subject_dirs:
            # Maybe they are in DATA_DIR/data/train ? Let's check DATA_DIR
            # Actually preprocessing saves to kaggle dataset/data/train/subject_01 etc.
            subject_dirs = sorted(glob.glob(os.path.join(config.DATA_DIR, "*", "sub-*"))) + \
                           sorted(glob.glob(os.path.join(config.DATA_DIR, "*", "subject_*")))
                           
        # Shuffle subject lists securely
        random.seed(42)
        shuffled_subjects = list(subject_dirs)
        random.shuffle(shuffled_subjects)
        
        if args.sanity_overfit:
            print("SANITY OVERFIT MODE: Using 1 subject for both training and validation.")
            train_subjects = shuffled_subjects[:1]
            val_subjects = train_subjects
        else:
            num_subjects = len(shuffled_subjects)
            train_count = int(0.8 * num_subjects)
            
            train_subjects = shuffled_subjects[:train_count]
            val_subjects = shuffled_subjects[train_count:]
        
        train_dataset = MRIMotionArtifactDataset(subject_dirs=train_subjects)
        val_dataset = MRIMotionArtifactDataset(subject_dirs=val_subjects)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    best_stage1_checkpoint_path = ""
    
    if args.stage in [1, 3]:
        print("Initializing Stage 1: Artifact Predictor...")
        if args.stage == 3:
            config.CHECKPOINT_DIR = os.path.join(base_checkpoint_dir, "stage1")
            config.setup_dirs()
            
        config.STAGE = 1
        model = Stage1ArtifactPredictor(base_filters=config.BASE_FILTERS_STAGE1).to(device)
        
        if args.multi_gpu:
            if torch.cuda.device_count() > 1:
                print("\n" + "="*40)
                print("GPU CONFIGURATION (STAGE 1)")
                print("="*40)
                print(f"CUDA available: {torch.cuda.is_available()}")
                print(f"Number of GPUs: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print("Multi-GPU enabled: True")
                print("Parallelization method: torch.nn.DataParallel")
                print("="*40 + "\n")
                model = torch.nn.DataParallel(model)
            else:
                print("Warning: --multi_gpu passed, but only 1 GPU found. Continuing with single GPU.")
                
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        
        trainer = Stage1Trainer(model, optimizer, train_loader, val_loader, config, device)
        best_stage1_checkpoint_path = trainer.train(resume_path=args.resume_checkpoint if args.resume_checkpoint else None)
        
    if args.stage in [2, 3]:
        print("Initializing Stage 2: Artifact-Guided Reconstructor...")
        if args.stage == 3:
            config.CHECKPOINT_DIR = os.path.join(base_checkpoint_dir, args.experiment_name)
            config.setup_dirs()
            args.stage1_checkpoint = best_stage1_checkpoint_path
            
        config.STAGE = 2
        if not args.stage1_checkpoint or not os.path.exists(args.stage1_checkpoint):
            print("Warning: No valid Stage 1 checkpoint provided. Using untrained Stage 1 model!")
            
        stage1_model = Stage1ArtifactPredictor(base_filters=config.BASE_FILTERS_STAGE1).to(device)
        if args.stage1_checkpoint and os.path.exists(args.stage1_checkpoint):
            checkpoint = torch.load(args.stage1_checkpoint, map_location=device)
            load_state_dict_flexible(stage1_model, checkpoint['state_dict'])
            print(f"Loaded Stage 1 checkpoint from {args.stage1_checkpoint}")
            
        stage2_model = Stage2ArtifactGuidedReconstructor(
            base_filters=config.BASE_FILTERS_STAGE2,
            num_residual_blocks=config.NUM_RESIDUAL_BLOCKS
        ).to(device)
        
        if args.finetune_checkpoint and os.path.exists(args.finetune_checkpoint):
            checkpoint2 = torch.load(args.finetune_checkpoint, map_location=device)
            load_state_dict_flexible(stage2_model, checkpoint2['state_dict'], safe_transfer=True)
            print(f"Loaded Stage-2 initialization checkpoint: {args.finetune_checkpoint}")
            print("Stage-2 initialized from existing trained weights. Mismatched shapes skipped.")
            
        if args.multi_gpu:
            if torch.cuda.device_count() > 1:
                print("\n" + "="*40)
                print("GPU CONFIGURATION")
                print("="*40)
                print(f"CUDA available: {torch.cuda.is_available()}")
                print(f"Number of GPUs: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print("Multi-GPU enabled: True")
                print("Parallelization method: torch.nn.DataParallel")
                print(f"Stage2 model wrapped in DataParallel: True")
                print(f"Input batch size: {config.BATCH_SIZE}")
                print(f"Expected samples per GPU: {max(1, config.BATCH_SIZE // torch.cuda.device_count())}")
                print("="*40 + "\n")
                stage1_model = torch.nn.DataParallel(stage1_model)
                stage2_model = torch.nn.DataParallel(stage2_model)
            else:
                print("Warning: --multi_gpu passed, but only 1 GPU found. Continuing with single GPU.")
            
        print("\n" + "="*50)
        print("STAGE 2 ARCHITECTURE & EQUATIONS")
        print("="*50)
        print("Stage-2 prediction type: Pure Residual Correction")
        print("Stage-2 reconstruction equation: Final = HM2 + PredictedCorrection")
        print("Loss components: Charbonnier, SSIM, FFL, Gradient Loss 3D, Artifact-Aware Preservation Loss")
        print("="*50 + "\n")
        
        optimizer = optim.Adam(stage2_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        
        trainer = Stage2Trainer(
            stage1_model, stage2_model, optimizer, train_loader, val_loader, config, device
        )
        
        if args.finetune_checkpoint:
            # We don't resume, we start fresh at epoch 0
            trainer.train(resume_path=None)
        else:
            trainer.train(resume_path=args.resume_checkpoint if args.resume_checkpoint else None)

if __name__ == "__main__":
    main()
