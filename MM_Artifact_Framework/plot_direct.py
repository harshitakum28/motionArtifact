import os
import numpy as np
import matplotlib.pyplot as plt

data_path = r"C:\Users\bitpa\OneDrive\Desktop\catest\data\train\subject_01\clean.npy"
print(f"Loading {data_path}...")
vol = np.load(data_path)
print(f"Volume shape: {vol.shape}")
print(f"Min: {vol.min()}, Max: {vol.max()}, Mean: {vol.mean()}")

slice_idx = vol.shape[2] // 2 # Z is the third dimension here? (196, 256, 256) (H, W, D). D is Z?
# In MRI, Axial is usually X-Y, and Z is the slice index.
# Let's plot slices along all 3 axes to see which one looks like a brain.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(vol[vol.shape[0]//2, :, :], cmap='gray')
axes[0].set_title("Slice along dim 0")

axes[1].imshow(vol[:, vol.shape[1]//2, :], cmap='gray')
axes[1].set_title("Slice along dim 1")

axes[2].imshow(vol[:, :, vol.shape[2]//2], cmap='gray')
axes[2].set_title("Slice along dim 2")

out_path = r"C:\Users\bitpa\OneDrive\Desktop\catest\data\visualizations\direct_plot.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path)
print(f"Saved direct plot to {out_path}")
