import os
import glob
import numpy as np

data_dir = r"C:\Users\bitpa\OneDrive\Desktop\catest\data\train"
subject_dirs = sorted(glob.glob(os.path.join(data_dir, "subject_*")))
print(f"Found {len(subject_dirs)} subjects")
if len(subject_dirs) > 0:
    subject_path = subject_dirs[0]
    print(f"Trying to load from {subject_path}")
    try:
        clean_vol = np.load(os.path.join(subject_path, 'clean.npy'))
        print("clean_vol shape:", clean_vol.shape)
    except Exception as e:
        print("Error loading:", repr(e))
