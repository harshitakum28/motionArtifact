import os
import torch

def upload_to_gdrive(filepath, folder_id):
    try:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        
        gauth = GoogleAuth()
        # Try to load saved client credentials
        gauth.LoadCredentialsFile("mycreds.txt")
        if gauth.credentials is None:
            # Authenticate if they're not there
            gauth.CommandLineAuth() 
        elif gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
        gauth.SaveCredentialsFile("mycreds.txt")
        
        drive = GoogleDrive(gauth)
        filename = os.path.basename(filepath)
        
        # Check if file exists, if so overwrite or create new
        file_list = drive.ListFile({'q': f"'{folder_id}' in parents and title='{filename}' and trashed=false"}).GetList()
        if len(file_list) > 0:
            file_drive = file_list[0]
        else:
            file_drive = drive.CreateFile({'title': filename, 'parents': [{'id': folder_id}]})
            
        file_drive.SetContentFile(filepath)
        file_drive.Upload()
        print(f"Successfully backed up {filename} to Google Drive folder {folder_id}.")
        
    except ImportError:
        print("PyDrive is not installed. Run '!pip install PyDrive' in Kaggle to enable Google Drive backup.")
    except Exception as e:
        error_msg = str(e)
        if "client_secrets.json" in error_msg or "InvalidConfigError" in error_msg:
            print(f"Google Drive Upload Failed: You need a 'client_secrets.json' file in your Kaggle working directory to authenticate PyDrive. Error: {e}")
        else:
            print(f"Failed to upload {os.path.basename(filepath)} to Google Drive. Are you running in a background Kaggle kernel where interactive auth is blocked? Error: {e}")

def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str, filename: str = 'checkpoint.pth', config=None, interval_epoch: int = None, is_window_best: bool = False, window_best_name: str = None):
    """
    Saves model checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_window_best and window_best_name:
        window_best_filepath = os.path.join(checkpoint_dir, window_best_name)
        torch.save(state, window_best_filepath)
    
    if interval_epoch is not None:
        interval_filepath = os.path.join(checkpoint_dir, f'epoch{interval_epoch}_{filename}')
        torch.save(state, interval_filepath)
        
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best_loss.pth')
        torch.save(state, best_filepath)
        
    if config and getattr(config, 'BACKUP_TO_GDRIVE', False):
        folder_id = getattr(config, 'GDRIVE_FOLDER_ID', None)
        if folder_id:
            upload_to_gdrive(filepath, folder_id)
            if interval_epoch is not None:
                upload_to_gdrive(interval_filepath, folder_id)
            if is_best:
                upload_to_gdrive(best_filepath, folder_id)
            if is_window_best and window_best_name:
                upload_to_gdrive(window_best_filepath, folder_id)

def load_state_dict_flexible(model: torch.nn.Module, state_dict: dict, safe_transfer: bool = False):
    """
    Safely loads a state_dict into a model, automatically handling the 'module.' 
    prefix introduced by DataParallel.
    If safe_transfer is True, it will skip mismatched shapes/keys and print a detailed report.
    """
    # 1. Clean the incoming state_dict by removing 'module.' prefix if it exists
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    # 2. Get the model's current state_dict
    target_model = model.module if hasattr(model, 'module') else model
    model_state = target_model.state_dict()
    
    if safe_transfer:
        loaded_keys = []
        skipped_keys = []
        final_state_dict = {}
        
        for name, param in model_state.items():
            if name in clean_state_dict:
                if clean_state_dict[name].shape == param.shape:
                    final_state_dict[name] = clean_state_dict[name]
                    loaded_keys.append(name)
                else:
                    skipped_keys.append((name, f"shape mismatch: ckpt {clean_state_dict[name].shape} vs model {param.shape}"))
                    final_state_dict[name] = param
            else:
                skipped_keys.append((name, "missing key in checkpoint"))
                final_state_dict[name] = param
                
        for name in clean_state_dict.keys():
            if name not in model_state:
                skipped_keys.append((name, "unexpected key in checkpoint (not in model)"))
                
        print(f"\n--- Checkpoint Transfer Report ---")
        print(f"Successfully loaded parameters: {len(loaded_keys)}")
        print(f"Skipped parameters: {len(skipped_keys)}")
        if skipped_keys:
            print("\nSkipped Keys Details:")
            for k, reason in skipped_keys:
                print(f" - {k}: {reason}")
        print("----------------------------------\n")
        
        target_model.load_state_dict(final_state_dict, strict=False)
    else:
        incompatible_keys = target_model.load_state_dict(clean_state_dict, strict=False)
        missing_keys = incompatible_keys.missing_keys
        unexpected_keys = incompatible_keys.unexpected_keys
        print(f"\n--- STRICT Checkpoint Transfer Report ---")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(f"Missing keys details: {missing_keys}")
            print(f"Unexpected keys details: {unexpected_keys}")
            print("-----------------------------------------\n")
            raise RuntimeError("Architecture mismatch detected during strict checkpoint loading!")
        print("-----------------------------------------\n")

def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer=None, safe_transfer: bool = False):
    """
    Loads model checkpoint.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")
        
    checkpoint = torch.load(filepath, map_location='cpu')
    load_state_dict_flexible(model, checkpoint['state_dict'], safe_transfer=safe_transfer)
    
    if optimizer and 'optimizer' in checkpoint and checkpoint['optimizer']:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
        
    return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))

def load_checkpoint_strict_for_inference(filepath: str, model: torch.nn.Module, device):
    """
    Robust checkpoint loader specifically designed for testing and inference.
    Automatically handles:
      - Wrapped checkpoints {"state_dict": ...} vs Raw state dictionaries.
      - DataParallel "module." prefixes.
    Enforces strict architectural compatibility and halts on failure.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    is_wrapped = False
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        is_wrapped = True
    else:
        state_dict = checkpoint
        
    # Detect DataParallel in checkpoint
    ckpt_dp = any(k.startswith('module.') for k in state_dict.keys())
    
    # Target underlying model
    model_dp = hasattr(model, 'module')
    target_model = model.module if model_dp else model
    
    # Strip prefix
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    expected_keys = set(target_model.state_dict().keys())
    provided_keys = set(clean_state_dict.keys())
    
    missing_keys = expected_keys - provided_keys
    unexpected_keys = provided_keys - expected_keys
    
    shape_mismatches = []
    for k in (expected_keys & provided_keys):
        expected_shape = target_model.state_dict()[k].shape
        provided_shape = clean_state_dict[k].shape
        if expected_shape != provided_shape:
            shape_mismatches.append((k, expected_shape, provided_shape))
            
    print("\n========================================")
    print("CHECKPOINT LOAD REPORT")
    print("========================================")
    print(f"Checkpoint: {filepath}")
    print(f"Checkpoint format: {'Wrapped state_dict' if is_wrapped else 'Raw state_dict'}")
    print(f"Checkpoint DataParallel prefix detected: {ckpt_dp}")
    print(f"Model DataParallel wrapper detected: {model_dp}")
    print(f"Parameters in checkpoint: {len(provided_keys)}")
    print(f"Parameters expected by model: {len(expected_keys)}")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    print(f"Shape mismatches: {len(shape_mismatches)}")
    
    if len(missing_keys) > 0 or len(unexpected_keys) > 0 or len(shape_mismatches) > 0:
        if len(missing_keys) > 0:
            print(f"\nMissing keys details: {list(missing_keys)[:10]} ...")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys details: {list(unexpected_keys)[:10]} ...")
        if len(shape_mismatches) > 0:
            print("Shape mismatch details:")
            for m in shape_mismatches:
                print(f"  {m[0]}: expected {m[1]}, got {m[2]}")
        print("========================================\n")
        raise RuntimeError(f"Strict architecture validation failed for checkpoint: {filepath}")
        
    print("========================================\n")
    
    # Load strictly
    target_model.load_state_dict(clean_state_dict, strict=True)
    return True

