import os
import h5py
from torch.utils.data import Dataset, DataLoader
import fastmri
from fastmri.data import transforms as T


class FastMRIClassificationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, center_crop_size=None):
        """
        Args:
            root_dir (str): Directory containing subfolders, e.g. "brain-train", "knee-train", etc.
            split (str): One of "train", "val", or "test".
            transform (callable, optional): Optional transform to apply to each sample.
            center_crop_size (int or tuple, optional): Size for center cropping. If int, crop will be (size, size); 
                                                       if tuple it should be (height, width).
        """
        self.root_dir = os.path.expanduser(root_dir)
        self.split = split.lower()
        self.transform = transform
        self.center_crop_size = center_crop_size

        self.samples = []  # list of (file_path, label)
        # Label 0 = "knee", 1 = "brain"
        for organ_label, organ_name in enumerate(['knee', 'brain']):
            organ_dir = os.path.join(self.root_dir, f"{organ_name}-{self.split}/")
            if not os.path.isdir(organ_dir):
                raise ValueError(f"Directory not found: {organ_dir}")

            candidate_dir = os.path.join(organ_dir, f"multicoil-{self.split}")
            if not os.path.isdir(candidate_dir):
                raise ValueError(f"Directory not found: {candidate_dir}")
            
            target_dir = candidate_dir

            for fname in os.listdir(target_dir):
                if fname.endswith('.h5'):
                    filepath = os.path.join(target_dir, fname)
                    self.samples.append((filepath, organ_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with h5py.File(file_path, 'r') as hf:
            kspace = hf['kspace'][()]  # shape: [num_slices, num_coils, H, W]

        center_slice_idx = kspace.shape[0] // 2
        kspace_slice = kspace[center_slice_idx]  # shape: [num_coils, H, W]

        # Convert to PyTorch tensor
        kspace_tensor = T.to_tensor(kspace_slice)

        # FastMRI-style reconstruction
        slice_image = fastmri.ifft2c(kspace_tensor)
        slice_image_abs = fastmri.complex_abs(slice_image)  # shape: [C, H, W]
        slice_image_rss = fastmri.rss(slice_image_abs, dim=0)  # shape: [H, W]

        # Add channel dimension: [1, H, W]
        image = slice_image_rss.unsqueeze(0)

        # Apply center crop if specified
        if self.center_crop_size is not None:
            image = T.center_crop(image, self.center_crop_size)

        return image, label

def main():
    print("Hello, FastMRI!")
    # Example: center crop to 128x128 pixels
    train_dataset = FastMRIClassificationDataset(
        root_dir="~/trial-project/data/", 
        split='train',
        center_crop_size=(128, 128)
    )
    val_dataset = FastMRIClassificationDataset(
        root_dir="~/trial-project/data/", 
        split='val',
        center_crop_size=(128, 128)
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    print("Number of samples in train dataset:", len(train_dataset))
    for batch in train_loader:
        images, labels = batch  # shape [B, 1, H, W] and labels
        print("Batch images shape:", images.shape)
        print("Batch images dtype:", images.dtype)
        print("Batch labels:", labels)
        break

if __name__ == '__main__':
    main()
