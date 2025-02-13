
from torch.utils.data import DataLoader
import fastmri
from fastmri.data import transforms as T

from fastmri.data.subsample import create_mask_for_mask_type
# custom transform function for SliceDataset

def custom_transform(kspace, mask, target, attrs, fname, slice_num):
    """
    Custom transform function that integrates `VarNetDataTransform` while still
    generating the cropped quick reconstruction image.
    """
    # Instantiate VarNetDataTransform
    varnet_transform = T.VarNetDataTransform(mask_func=create_mask_for_mask_type('random', [0.08], [4]))
    
    # Process k-space using VarNetDataTransform
    varnet_sample = varnet_transform(kspace, mask, target, attrs, fname, slice_num)
    masked_kspace = varnet_sample.masked_kspace
    mask = varnet_sample.mask
    target = varnet_sample.target
    
    # Compute quick reconstruction for the encoder
    quick_recon_image = fastmri.ifft2c(masked_kspace)
    quick_recon_abs = fastmri.complex_abs(quick_recon_image)
    quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)  # Shape [1, H, W]
    quick_recon_rss = T.center_crop(quick_recon_rss, (128, 128))
    
    return masked_kspace, mask, quick_recon_rss, target, fname, slice_num
    



def main():
    # print("Hello, FastMRI!")

    from mri_data_custom import SliceDataset

    train_dataset = SliceDataset(
        root="/Users/ericq/trial-project/data/knee-train/multicoil-train/",
        challenge="multicoil",
        transform=custom_transform,
    )

    # print("finished creating dataset")

    # Create DataLoader
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Test the loader
    for batch in dataloader:
        masked_kspace, mask, quick_recon_rss, target, fname, slice_num = batch
        print(f"Masked K-space shape: {masked_kspace.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Quick Reconstruction shape: {quick_recon_rss.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Filenames: {fname}")
        print(f"Slice numbers: {slice_num}")
        break

if __name__ == '__main__':
    main()