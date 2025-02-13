import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = '/Users/ericq/trial-project/data/brain-train/multicoil-train/file_brain_AXT2_210_6001944.h5'

with h5py.File(file_path, 'r') as hf:
    ground_truth = hf['reconstruction_rss'][()]  # Ground truth
    kspace = hf['kspace'][()]  # Raw k-space data
    # mask = hf['mask'][()]  # Sampling mask
    
    # Pick the center slice
    center_slice_idx = kspace.shape[0] // 2
    kspace_slice = kspace[center_slice_idx]  # [num_coils, H, W]
    # mask_slice = mask[center_slice_idx]  # Mask corresponding to the slice
    target_slice = ground_truth[center_slice_idx]  # Ground truth image

# Display the k-space and ground truth image
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow((np.log(np.abs(kspace_slice)))[5], cmap=None)
plt.title("k-space (log scale)")
plt.axis('off')

plt.subplot(122)
plt.imshow(target_slice, cmap=None)
plt.title("Ground truth image")
plt.axis('off')

plt.show()