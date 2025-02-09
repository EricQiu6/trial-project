import os
import torch
import h5py
from torch.utils.data import Dataset
import fastmri
from fastmri.data import transforms as T
from torch.utils.data import DataLoader


class FastMRIClassificationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to a directory that contains subfolders, e.g. "brain-train", "knee-train", etc.
            split (str): One of "train", "val", or "test".
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.root_dir = root_dir
        self.split = split.lower()
        self.transform = transform

        # Collect all .h5 file paths in the provided structure
        self.samples = []  # list of (file_path, label)

        # Label 0 = "knee", 1 = "brain"
        for organ_label, organ_name in enumerate(['knee', 'brain']):
            # Construct expected folder name, e.g., "knee-train", "brain-test", etc.
            organ_dir = os.path.join(root_dir, f"{organ_name}-{self.split}")
            if not os.path.isdir(organ_dir):
                continue  # Skip if folder doesn't exist

            # For train, check if there's an extra subfolder (e.g., "multicoil_train")
            candidate_dir = os.path.join(organ_dir, "multicoil_train")
            if os.path.isdir(candidate_dir):
                target_dir = candidate_dir
            else:
                target_dir = organ_dir

            for fname in os.listdir(target_dir):
                if fname.endswith('.h5'):
                    filepath = os.path.join(target_dir, fname)
                    self.samples.append((filepath, organ_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - 'image': torch.FloatTensor of shape [1, H, W]
                - 'label': int (0 for knee, 1 for brain)
        """
        file_path, label = self.samples[idx]

        with h5py.File(file_path, 'r') as hf:
            # The fastMRI data typically has shape [num_slices, num_coils, height, width]
            kspace = hf['kspace'][()]  # e.g. shape [S, C, H, W]

        # Pick the center slice
        center_slice_idx = kspace.shape[0] // 2
        kspace_slice = kspace[center_slice_idx]  # shape [num_coils, H, W]

        # Convert to PyTorch tensor
        kspace_slice2 = T.to_tensor(kspace_slice)

        # --- FastMRI-style reconstruction ---

        slice_image = fastmri.ifft2c(kspace_slice2) # Apply Inverse Fourier Transform to get the complex image

        slice_image_abs = fastmri.complex_abs(slice_image)  # [C, H, W]

        # 3) Root-sum-of-squares coil combination
        slice_image_rss = fastmri.rss(slice_image_abs, dim=0)  # [H, W]

        # Add channel dimension: [1, H, W]
        image = slice_image_rss.unsqueeze(0)

        sample = {
            'image': image,
            'label': label
        }

        # Optional: apply transforms (e.g., normalization, augmentations)
        if self.transform:
            sample = self.transform(sample)

        return sample
    
train_dataset = FastMRIClassificationDataset(root_dir="fastMRI_classification", split='train')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

for batch in train_loader:
    images = batch['image']  # shape [B, 1, H, W]
    labels = batch['label']  # shape [B]
    print("Batch images shape:", images.shape)
    print("Batch labels:", labels)
    break