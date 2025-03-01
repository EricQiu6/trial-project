import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import create_mask_for_mask_type
from data_module_custom import FastMriDataModule
from fastmri.data.transforms import to_tensor, complex_center_crop
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import sys
import traceback

# imported from fastmri banding removal
# def coil_compress(kspace, out_coils):
#     if kspace.shape[0] <= out_coils:
#         print("kspace coil num is less: ", kspace.shape[0])
#         return kspace

#     kspace = kspace[..., 0] + 1j * kspace[..., 1]

#     start_shape = tuple(kspace.shape)
#     in_coils = start_shape[0]
#     kspace = kspace.reshape(in_coils, -1)
#     try:
#         if in_coils == 5:
#             u, _, _ = svd(kspace, full_matrices=False)
#         else:
#             u, _, _ = svds(kspace, k=out_coils)
#     except Exception as e:
#         print("SVD failed: ", kspace.shape)
#         traceback.print_exc(file=sys.stdout)
#         raise e

#     u = np.transpose(np.conj(u[:, :out_coils]))
#     new_shape = (out_coils, ) + start_shape[1:]
#     new_kspace = u @ kspace
#     kspace = np.reshape(new_kspace, new_shape)

#     kspace = torch.stack((torch.Tensor(np.real(kspace)), torch.Tensor(np.imag(kspace))), dim=-1)
#     return kspace

def ifft_crop_fft(kspace: torch.Tensor, crop_size=(320, 320)) -> torch.Tensor:
    """
    Crops k-space to `crop_size` in the spatial domain by:
      1) Inverse FFT from k-space to image space
      2) Center-crop the image
      3) FFT back to k-space

    The padding is applied to the third-to-last and second-to-last dimensions (i.e. the spatial H and W dimensions)
    
    Args:
        kspace: [C, H, W, 2] complex k-space (e.g. multi-coil if C > 1).
        crop_size: (height, width) to crop in image space.
    
    Returns:
        A new k-space tensor [C, crop_h, crop_w, 2].
    """
    # 1) Go to image space
    image = fastmri.ifft2c(kspace)  # [C, H, W, 2] -> image domain
    # print("image shape in ifft:", image.shape) 

    # Calculate necessary padding for height and width dimensions
    pad_h = max(crop_size[0] - image.shape[-3], 0)
    pad_w = max(crop_size[1] - image.shape[-2], 0)

    # Apply padding to third-to-last (height) and second-to-last (width) dimensions
    # Pad tuple format for 4D tensor with 3 pairs: (last_dim_left, last_dim_right,
    # second_last_left, second_last_right, third_last_left, third_last_right)
    # Here, no padding is added to the last dimension.
    image = F.pad(
        image,
        (0, 0,              # No padding for the last dimension (complex components)
         pad_w // 2, pad_w - pad_w // 2,  # Pad width dimension (second-to-last)
         pad_h // 2, pad_h - pad_h // 2)  # Pad height dimension (third-to-last)
    )
    # print("image shape after padding:", image.shape)

    # 2) Center-crop the image using the provided crop size
    image_cropped = complex_center_crop(image, crop_size)
    # 3) Transform back to k-space
    kspace_cropped = fastmri.fft2c(image_cropped)
    return kspace_cropped

def handle_coil_variability(kspace, num_coils=8):
    """
    Ensures k-space has a fixed number of coils and converts it to a tensor.
    """
    kspace = to_tensor(kspace)  # Convert to tensor if it's not already
    num_coils_available = kspace.shape[0]
    if num_coils_available >= num_coils:
        return kspace[:num_coils]  # Truncate extra coils
    else:
        pad_size = num_coils - num_coils_available
        return torch.cat([kspace, torch.zeros((pad_size, *kspace.shape[1:]), dtype=kspace.dtype, device=kspace.device)], dim=0)

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

# use this one for sensitivity maps computation
def custom_transform_combine(kspace, mask, target, attrs, fname, slice_num):

  if target is not None:
      target_torch = to_tensor(target)
      max_value = attrs["max"]
  else:
      print("target is None, reconstructing...")
      quick_recon_image = fastmri.ifft2c(masked_kspace)
      quick_recon_abs = fastmri.complex_abs(quick_recon_image)
      quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)
      target_torch = quick_recon_rss
      max_value = target_torch.max()

  kspace_torch = handle_coil_variability(kspace, 10)
  # if kspace is not a tensor, convert it to a tensor
  if not isinstance(kspace_torch, torch.Tensor):
    kspace_torch = to_tensor(kspace)

  mask_func = create_mask_for_mask_type('random', [0.08], [4])
  use_seed = True

  seed = None if not use_seed else tuple(map(ord, fname))
  acq_start = attrs["padding_left"]
  acq_end = attrs["padding_right"]

  # crop_size_attr = (attrs["recon_size"][0], attrs["recon_size"][1])
  # print("crop size attr: ", crop_size_attr)
  # kspace_torch = center_crop_kspace(kspace_torch, crop_size_attr)
  # print("kspace shape after attr crop: ", kspace_torch.shape)

  crop_size_centering = (320, 320)
  kspace_torch = ifft_crop_fft(kspace_torch, crop_size_centering)
  print("kspace shape after centering crop: ", kspace_torch.shape)

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
          crop_size=crop_size_centering,
      )
  else:
      print("mask_func is None!")
      exit(1)

  masked_kspace = sample.masked_kspace
  mask = sample.mask
  target = sample.target

  # Compute quick reconstruction for the encoder
  try:
    quick_recon_image = fastmri.ifft2c(masked_kspace)
    quick_recon_abs = fastmri.complex_abs(quick_recon_image)
    quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)  # Shape [1, H, W]
  except (RuntimeError, ValueError) as e:
        print("Invalid shapes encountered:", e)
        print(f"quick_recon_rss shape: {quick_recon_rss.shape if 'quick_recon_rss' in locals() else 'N/A'}")
        exit(1)

  return masked_kspace, mask, quick_recon_rss, target, fname, slice_num

def custom_transform_combine_train(kspace, mask, target, attrs, fname, slice_num):
  if target is not None:
      target_torch = to_tensor(target)
      max_value = attrs["max"]
  else:
      print("target is None, reconstructing...")
      quick_recon_image = fastmri.ifft2c(masked_kspace)
      quick_recon_abs = fastmri.complex_abs(quick_recon_image)
      quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)
      target_torch = quick_recon_rss
      max_value = target_torch.max()

  kspace_torch = handle_coil_variability(kspace, 10)
  # if kspace is not a tensor, convert it to a tensor
  if not isinstance(kspace_torch, torch.Tensor):
    kspace_torch = to_tensor(kspace)

  mask_func = create_mask_for_mask_type('random', [0.08], [4])
  use_seed = False # use random seed for val

  seed = None if not use_seed else tuple(map(ord, fname))
  acq_start = attrs["padding_left"]
  acq_end = attrs["padding_right"]

  # crop_size_attr = (attrs["recon_size"][0], attrs["recon_size"][1])
  # print("crop size attr: ", crop_size_attr)
  # kspace_torch = center_crop_kspace(kspace_torch, crop_size_attr)
  # print("kspace shape after attr crop: ", kspace_torch.shape)

  crop_size_centering = (320, 320)
  kspace_torch = ifft_crop_fft(kspace_torch, crop_size_centering)
  print("kspace shape after centering crop: ", kspace_torch.shape)

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
          crop_size=crop_size_centering,
      )
  else:
      print("mask_func is None!")
      exit(1)

  masked_kspace = sample.masked_kspace
  mask = sample.mask
  target = sample.target

  # Compute quick reconstruction for the encoder
  try:
    quick_recon_image = fastmri.ifft2c(masked_kspace)
    quick_recon_abs = fastmri.complex_abs(quick_recon_image)
    quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)  # Shape [1, H, W]
  except (RuntimeError, ValueError) as e:
        print("Invalid shapes encountered:", e)
        print(f"quick_recon_rss shape: {quick_recon_rss.shape if 'quick_recon_rss' in locals() else 'N/A'}")
        exit(1)

  # has precomputed sensitivity maps in train/val
  fname_stem = Path(fname).stem
  save_dir = Path(f"sens_maps_ifft_crop_fft/train/{fname_stem}")  # Adjust split as needed
  sens_map_path = save_dir / f"sens_map_slice{slice_num}.pt"
  sens_maps = torch.load(sens_map_path)

  if torch.isnan(sens_maps).any():
        print("ERROR: `sens_maps` when loaded contains NaNs!")
        exit(1)


  return masked_kspace, mask, quick_recon_rss, target, fname, slice_num, sens_maps

def custom_transform_combine_val(kspace, mask, target, attrs, fname, slice_num):
  if target is not None:
      target_torch = to_tensor(target)
      max_value = attrs["max"]
  else:
      print("target is None, reconstructing...")
      quick_recon_image = fastmri.ifft2c(masked_kspace)
      quick_recon_abs = fastmri.complex_abs(quick_recon_image)
      quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)
      target_torch = quick_recon_rss
      max_value = target_torch.max()

  kspace_torch = handle_coil_variability(kspace, 10)
  # if kspace is not a tensor, convert it to a tensor
  if not isinstance(kspace_torch, torch.Tensor):
    kspace_torch = to_tensor(kspace)

  mask_func = create_mask_for_mask_type('random', [0.08], [4])
  use_seed = True # use seed for val

  seed = None if not use_seed else tuple(map(ord, fname))
  acq_start = attrs["padding_left"]
  acq_end = attrs["padding_right"]

  # crop_size_attr = (attrs["recon_size"][0], attrs["recon_size"][1])
  # print("crop size attr: ", crop_size_attr)
  # kspace_torch = center_crop_kspace(kspace_torch, crop_size_attr)
  # print("kspace shape after attr crop: ", kspace_torch.shape)

  crop_size_centering = (320, 320)
  kspace_torch = ifft_crop_fft(kspace_torch, crop_size_centering)
  print("kspace shape after centering crop: ", kspace_torch.shape)

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
          crop_size=crop_size_centering,
      )
  else:
      print("mask_func is None!")
      exit(1)

  masked_kspace = sample.masked_kspace
  mask = sample.mask
  target = sample.target

  # Compute quick reconstruction for the encoder
  try:
    quick_recon_image = fastmri.ifft2c(masked_kspace)
    quick_recon_abs = fastmri.complex_abs(quick_recon_image)
    quick_recon_rss = fastmri.rss(quick_recon_abs, dim=0).unsqueeze(0)  # Shape [1, H, W]
  except (RuntimeError, ValueError) as e:
        print("Invalid shapes encountered:", e)
        print(f"quick_recon_rss shape: {quick_recon_rss.shape if 'quick_recon_rss' in locals() else 'N/A'}")
        exit(1)

  # has precomputed sensitivity maps in train/val
  fname_stem = Path(fname).stem
  save_dir = Path(f"sens_maps_ifft_crop_fft/val/{fname_stem}")  # Adjust split as needed
  sens_map_path = save_dir / f"sens_map_slice{slice_num}.pt"
  sens_maps = torch.load(sens_map_path)

  if torch.isnan(sens_maps).any():
        print("ERROR: `sens_maps` when loaded contains NaNs!")
        exit(1)


  return masked_kspace, mask, quick_recon_rss, target, fname, slice_num, sens_maps

def main():
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
  plt.savefig("datasetvult_visualized.png")
  print("saved figure to datasetvult_visualized.png")


if __name__ == "__main__":
  main()