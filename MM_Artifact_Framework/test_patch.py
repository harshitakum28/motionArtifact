import os
import numpy as np
import matplotlib.pyplot as plt

data_path = r"C:\Users\bitpa\OneDrive\Desktop\catest\data\train\subject_01\clean.npy"
vol = np.load(data_path)

# Extract middle patch (patch 8 out of 16)
patch_depth = 256 // 16
start_idx = 8 * patch_depth
end_idx = 9 * patch_depth
patch = vol[:, :, start_idx:end_idx]

# Normalize per patch
patch_norm = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)

# Plot middle slice of the patch
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(patch_norm[:, :, 8], cmap='gray')
ax.set_title("Middle Patch - Middle Slice")

out_path = r"C:\Users\bitpa\OneDrive\Desktop\catest\data\visualizations\middle_patch.png"
plt.savefig(out_path)
print(f"Saved middle patch to {out_path}")
