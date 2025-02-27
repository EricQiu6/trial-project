import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import create_mask_for_mask_type
from data_module_custom import FastMriDataModule
from fastmri.data.transforms import center_crop

def ifft_crop_fft(kspace: torch.Tensor, crop_size=(320, 320)) -> torch.Tensor:
    """
    Crops k-space to `crop_size` in the spatial domain by:
      1) Inverse FFT from k-space to image space
      2) Center-crop the image
      3) FFT back to k-space
    
    Args:
        kspace: [C, H, W, 2] complex k-space (e.g. multi-coil if C > 1).
        crop_size: (height, width) to crop in image space.
    
    Returns:
        A new k-space tensor [C, crop_h, crop_w, 2].
    """
    # 1) Go to image space
    image = fastmri.ifft2c(kspace)  # [C, H, W, 2] -> image domain
    print("image shape in ifft:", image.shape) 
    quick_recon_abs = fastmri.complex_abs(image)
    quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0)
    # 2) Center-crop the image
    image_cropped = center_crop(quick_recon_rss, crop_size)
    # 3) Transform back to k-space
    kspace_cropped = fastmri.fft2c(image_cropped)
    return kspace_cropped

# use this one for sensitivity maps computation
def custom_transform_combine(kspace, mask, target, attrs, fname, slice_num):

    kspace = handle_coil_variability(kspace)

    mask_func = create_mask_for_mask_type('random', [0.08], [4])
    use_seed = True

    if target is not None:
        target_torch = T.to_tensor(target)
        max_value = attrs["max"]
    else:
        target_torch = torch.tensor(0)
        max_value = 0.0

    crop_size = (320, 320)

    # print(f"K-space shape before: {kspace.shape}")
    kspace_torch = T.to_tensor(kspace)
    # print(f"K-space shape torch: {kspace_torch.shape}")
    kspace_torch = ifft_crop_fft(kspace_torch, crop_size)
    # print(f"K-space shape after: {kspace_torch.shape}")
    seed = None if not use_seed else tuple(map(ord, fname))
    acq_start = attrs["padding_left"]
    acq_end = attrs["padding_right"]

    if mask_func is not None:
        masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
            kspace_torch, mask_func, seed=seed, padding=(acq_start, acq_end)
        )

        sample = T.VarNetSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=num_low_frequencies,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
        )
    else:
        masked_kspace = kspace_torch
        shape = np.array(kspace_torch.shape)
        num_cols = shape[-2]
        shape[:-3] = 1
        mask_shape = [1] * len(shape)
        mask_shape[-2] = num_cols
        mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        mask_torch = mask_torch.reshape(*mask_shape)
        mask_torch[:, :, :acq_start] = 0
        mask_torch[:, :, acq_end:] = 0

        sample = T.VarNetSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=0,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
        )

    masked_kspace = sample.masked_kspace
    mask = sample.mask
    target = sample.target

    # Compute quick reconstruction for the encoder
    quick_recon_image = fastmri.ifft2c(masked_kspace)
    quick_recon_abs = fastmri.complex_abs(quick_recon_image)
    quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)  # Shape [1, H, W]
    quick_recon_rss = T.center_crop(quick_recon_rss, (128, 128))
    
    return masked_kspace, mask, quick_recon_rss, target, fname, slice_num

def custom_transform_combine_train(kspace, mask, target, attrs, fname, slice_num):

    kspace = handle_coil_variability(kspace)

    mask_func = create_mask_for_mask_type('random', [0.08], [4])
    use_seed = True

    if target is not None:
        target_torch = T.to_tensor(target)
        max_value = attrs["max"]
    else:
        target_torch = torch.tensor(0)
        max_value = 0.0

    crop_size = (320, 320)

    # print(f"K-space shape before: {kspace.shape}")
    kspace_torch = T.to_tensor(kspace)
    # print(f"K-space shape torch: {kspace_torch.shape}")
    kspace_torch = ifft_crop_fft(kspace_torch, crop_size)
    # print(f"K-space shape after: {kspace_torch.shape}")
    seed = None if not use_seed else tuple(map(ord, fname))
    acq_start = attrs["padding_left"]
    acq_end = attrs["padding_right"]

    if mask_func is not None:
        masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
            kspace_torch, mask_func, seed=seed, padding=(acq_start, acq_end)
        )

        sample = T.VarNetSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=num_low_frequencies,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
        )
    else:
        masked_kspace = kspace_torch
        shape = np.array(kspace_torch.shape)
        num_cols = shape[-2]
        shape[:-3] = 1
        mask_shape = [1] * len(shape)
        mask_shape[-2] = num_cols
        mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        mask_torch = mask_torch.reshape(*mask_shape)
        mask_torch[:, :, :acq_start] = 0
        mask_torch[:, :, acq_end:] = 0

        sample = T.VarNetSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=0,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
        )

    masked_kspace = sample.masked_kspace
    mask = sample.mask
    target = sample.target

    try:
        # Compute quick reconstruction for the encoder
        quick_recon_image = fastmri.ifft2c(masked_kspace)
        quick_recon_abs = fastmri.complex_abs(quick_recon_image)
        quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)  # Shape [1, H, W]
        quick_recon_rss = T.center_crop(quick_recon_rss, (128, 128))
    except (RuntimeError, ValueError) as e:
        print("Invalid shapes encountered:", e)
        print(f"quick_recon_rss shape: {quick_recon_rss.shape if 'quick_recon_rss' in locals() else 'N/A'}")
        exit(1)

    # Load precomputed sensitivity maps
    fname_stem = Path(fname).stem
    save_dir = Path(f"sens_maps_no_weird_kspace_trans/train/{fname_stem}")  # Adjust split as needed
    sens_map_path = save_dir / f"sens_map_slice{slice_num}.pt"
    # print(f"Loading sensitivity map from: {sens_map_path}")
    sens_maps = torch.load(sens_map_path)

    if torch.isnan(sens_maps).any():
        print("ERROR: `sens_maps` when loaded contains NaNs!")
        exit(1)
    
    return masked_kspace, mask, quick_recon_rss, target, fname, slice_num, sens_maps

def custom_transform_combine_val(kspace, mask, target, attrs, fname, slice_num):

    kspace = handle_coil_variability(kspace)

    mask_func = create_mask_for_mask_type('random', [0.08], [4])
    use_seed = True

    if target is not None:
        target_torch = T.to_tensor(target)
        max_value = attrs["max"]
    else:
        target_torch = torch.tensor(0)
        max_value = 0.0

    crop_size = (320, 320)

    # print(f"K-space shape before: {kspace.shape}")
    kspace_torch = T.to_tensor(kspace)
    # print(f"K-space shape torch: {kspace_torch.shape}")
    kspace_torch = ifft_crop_fft(kspace_torch, crop_size)
    # print(f"K-space shape after: {kspace_torch.shape}")
    seed = None if not use_seed else tuple(map(ord, fname))
    acq_start = attrs["padding_left"]
    acq_end = attrs["padding_right"]

    if mask_func is not None:
        masked_kspace, mask_torch, num_low_frequencies = T.apply_mask(
            kspace_torch, mask_func, seed=seed, padding=(acq_start, acq_end)
        )

        sample = T.VarNetSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=num_low_frequencies,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
        )
    else:
        masked_kspace = kspace_torch
        shape = np.array(kspace_torch.shape)
        num_cols = shape[-2]
        shape[:-3] = 1
        mask_shape = [1] * len(shape)
        mask_shape[-2] = num_cols
        mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        mask_torch = mask_torch.reshape(*mask_shape)
        mask_torch[:, :, :acq_start] = 0
        mask_torch[:, :, acq_end:] = 0

        sample = T.VarNetSample(
            masked_kspace=masked_kspace,
            mask=mask_torch.to(torch.bool),
            num_low_frequencies=0,
            target=target_torch,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=crop_size,
        )

    masked_kspace = sample.masked_kspace
    mask = sample.mask
    target = sample.target

    try:
        # Compute quick reconstruction for the encoder
        quick_recon_image = fastmri.ifft2c(masked_kspace)
        quick_recon_abs = fastmri.complex_abs(quick_recon_image)
        quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)  # Shape [1, H, W]
        # quick_recon_rss = T.center_crop(quick_recon_rss, (128, 128))
    except (RuntimeError, ValueError) as e:
        print("Invalid shapes encountered:", e)
        print(f"quick_recon_rss shape: {quick_recon_rss.shape if 'quick_recon_rss' in locals() else 'N/A'}")
        exit(1)

    # Load precomputed sensitivity maps
    fname_stem = Path(fname).stem
    save_dir = Path(f"sens_maps_no_weird_kspace_trans/val/{fname_stem}")  # Adjust split as needed
    sens_map_path = save_dir / f"sens_map_slice{slice_num}.pt"
    sens_maps = torch.load(sens_map_path)
    
    return masked_kspace, mask, quick_recon_rss, target, fname, slice_num, sens_maps



import torch.nn.functional as F

def center_crop_kspace(tensor, crop_size):
    """
    Args:
        tensor: Input k-space [C, H, W, 2]
        crop_size: Desired spatial crop (H, W)
    """
    _, h, w, _ = tensor.shape
    crop_h, crop_w = crop_size

    # Pad height and width if needed (not the complex dimension!)
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)

    # PyTorch F.pad order: (left, right, top, bottom) for last two dimensions
    # We need to pad (H, W), so use (top, bottom, left, right)
    if pad_h > 0 or pad_w > 0:
        print("right before padding:" + str(tensor.shape))
        tensor = F.pad(
            tensor,
            (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        )
        print("right after padding:" + str(tensor.shape))

    # Now crop
    start_h = (tensor.shape[1] - crop_h) // 2
    start_w = (tensor.shape[2] - crop_w) // 2
    return tensor[:, start_h : start_h + crop_h, start_w : start_w + crop_w, :]

def handle_coil_variability(kspace, num_coils=8):
    """
    Ensures k-space has a fixed number of coils and converts it to a tensor.
    """
    kspace = torch.tensor(kspace)  # Convert to tensor if it's not already
    num_coils_available = kspace.shape[0]
    if num_coils_available >= num_coils:
        return kspace[:num_coils]  # Truncate extra coils
    else:
        pad_size = num_coils - num_coils_available
        return torch.cat([kspace, torch.zeros((pad_size, *kspace.shape[1:]), dtype=kspace.dtype, device=kspace.device)], dim=0)

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
    
    # print(f"Masked K-space shape before: {masked_kspace.shape}")
    masked_kspace_ifft = fastmri.ifft2c(masked_kspace)
    masked_kspace_cropped = center_crop_kspace(masked_kspace_ifft, (320, 320))
    masked_kspace = fastmri.fft2c(masked_kspace_cropped)
    # print(f"Masked K-space shape after: {masked_kspace.shape}")

    # Compute quick reconstruction for the encoder
    quick_recon_image = fastmri.ifft2c(masked_kspace)
    quick_recon_abs = fastmri.complex_abs(quick_recon_image)
    quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)  # Shape [1, H, W]
    quick_recon_rss = T.center_crop(quick_recon_rss, (128, 128))

    # # Ensure tensors are cloned so that they own their memory
    # masked_kspace = masked_kspace.clone()
    # mask = mask.clone()
    # quick_recon_rss = quick_recon_rss.clone()
    # target = target.clone() if isinstance(target, torch.Tensor) else target
    
    return masked_kspace, mask, quick_recon_rss, target, fname, slice_num
    



def main():
    # # Instantiate FastMriDataModule
    # data_module = FastMriDataModule(
    #     data_path=Path("/Users/ericq/trial-project/official-fitting-data/brain/"),
    #     challenge="multicoil",
    #     train_transform=custom_transform_combine,
    #     val_transform=custom_transform_combine,
    #     test_transform=custom_transform_combine,
    #     batch_size=4,
    #     num_workers=2,
    # )

    # # Prepare data (if needed)
    # data_module.prepare_data()

    # # Get train dataloader
    # train_dataloader = data_module.train_dataloader()

    # # Test the loader
    # for batch in train_dataloader:
    #     masked_kspace, mask, quick_recon_rss, target, fname, slice_num = batch
    #     print(f"Masked K-space shape: {masked_kspace.shape}")
    #     print(f"Mask shape: {mask.shape}")
    #     print(f"Quick Reconstruction shape: {quick_recon_rss.shape}")
    #     print(f"Target shape: {target.shape}")
    #     print(f"Filenames: {fname}")
    #     print(f"Slice numbers: {slice_num}")

    # data_module_brain = FastMriDataModule(
    #     data_path=Path("/Users/ericq/trial-project/official-fitting-data/brain/"),
    #     challenge="multicoil",
    #     train_transform=custom_transform,
    #     val_transform=custom_transform,
    #     test_transform=custom_transform,
    #     batch_size=4,
    #     num_workers=2,
    # )

    # # Prepare data (if needed)
    # data_module_brain.prepare_data()

    # # Get train dataloader
    # train_dataloader_brain = data_module_brain.train_dataloader()

    # # Test the loader
    # for batch in train_dataloader_brain:
    #     masked_kspace, mask, quick_recon_rss, target, fname, slice_num = batch
    #     print(f"Masked K-space shape: {masked_kspace.shape}")
    #     print(f"Mask shape: {mask.shape}")
    #     print(f"Quick Reconstruction shape: {quick_recon_rss.shape}")
    #     print(f"Target shape: {target.shape}")
    #     print(f"Filenames: {fname}")
    #     print(f"Slice numbers: {slice_num}")

    #     break

    data_paths = [
        Path("/home/sq225/trial-project/data/brain-train/multicoil-train"),
        Path("/home/sq225/trial-project/data/knee-train/multicoil-train")
    ]

    data_module_combined = FastMriDataModule(
        data_path=Path("fake bro"),
        challenge="multicoil",
        train_transform=custom_transform_combine_train,
        val_transform=custom_transform_combine_val,
        test_transform=custom_transform_combine_val,
        batch_size=1,
        num_workers=1,
        combine_diff_organs=True,
        data_paths_for_combine=data_paths
    )

    # # Prepare data (if needed)
    # data_module_combined.prepare_data()

    # Get train dataloader
    train_dataloader_combined = data_module_combined.train_dataloader()

    # extract a batch and visualize masked kspace, quick_recon_rss, and target
    import matplotlib.pyplot as plt
    import numpy as np

    sample = next(iter(train_dataloader_combined))
    masked_kspace, mask, quick_recon_rss, target, fname, slice_num, sens_maps = sample

    # For visualization, we assume batch size = 1.
    # Visualize the magnitude of the k-space data from the first coil.
    mk = masked_kspace[0]  # shape: [coils, H, W, 2]
    coil0 = mk[0]  # take first coil -> shape: [H, W, 2]
    coil0_complex = coil0[..., 0] + 1j * coil0[..., 1]
    coil0_abs = np.abs(coil0_complex)

    # Convert quick_recon_rss and target to numpy arrays.
    # quick_recon_rss is expected to be a tensor of shape [1, H, W]
    print("quick_recon_rss shape: ", quick_recon_rss.shape)
    print("target shape: ", target.shape)
    qr = quick_recon_rss[0][0].detach().cpu().numpy()
    tgt = target[0].detach().cpu().numpy()

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Masked K-space (Coil 0 Magnitude, log-scale)")
    plt.imshow(np.log(coil0_abs + 1e-6), cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Quick Reconstruction RSS (for encoder)")
    plt.imshow(qr, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Target")
    plt.imshow(tgt, cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    # save the figure
    plt.savefig("datasetv2_visualized.png")
    

    # # Test the loader
    # for batch in train_dataloader_combined:
    #     masked_kspace, mask, quick_recon_rss, target, fname, slice_num = batch
    #     print(f"Masked K-space shape: {masked_kspace.shape}")
    #     print(f"Mask shape: {mask.shape}")
    #     print(f"Quick Reconstruction shape: {quick_recon_rss.shape}")
    #     print(f"Target shape: {target.shape}")
    #     print(f"Filenames: {fname}")
    #     print(f"Slice numbers: {slice_num}")

    # data_module_mixed = FastMriDataModule(
    #     data_path=Path("/Users/ericq/trial-project/official-fitting-data/mix/"),
    #     challenge="multicoil",
    #     train_transform=custom_transform,
    #     val_transform=custom_transform,
    #     test_transform=custom_transform,
    #     batch_size=4,
    #     num_workers=2,
    # )

    # # Prepare data (if needed)
    # data_module_mixed.prepare_data()

    # # Get train dataloader
    # train_dataloader_mixed = data_module_mixed.train_dataloader()

    # # Test the loader
    # for batch in train_dataloader_mixed:
    #     masked_kspace, mask, quick_recon_rss, target, fname, slice_num = batch
    #     print(f"Masked K-space shape: {masked_kspace.shape}")
    #     print(f"Mask shape: {mask.shape}")
    #     print(f"Quick Reconstruction shape: {quick_recon_rss.shape}")
    #     print(f"Target shape: {target.shape}")
    #     print(f"Filenames: {fname}")
    #     print(f"Slice numbers: {slice_num}")

    #     break

if __name__ == '__main__':
    main()