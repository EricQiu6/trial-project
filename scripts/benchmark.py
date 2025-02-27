import torch
import torch.nn.functional as F
import numpy as np
import os

from pathlib import Path
from fastmri.losses import SSIMLoss
from fastmri.data import transforms
from varnet_module_vds import LatentVarNet as MyVarNet  # your 1-cascade version
from fastmri.models import VarNet as OfficialVarNet  # official 12-cascade version
import fastmri.evaluate as evaluate
from varnet_module_vds import VarNetBlock, NormUnet  # or define a 12-cascade version if needed
from data_module_custom import FastMriDataModule
from datasetv2 import custom_transform_combine_val
from utils import load_checkpoint, save_checkpoint, load_model

# ----------------------------------------------------------
# 1) Datasets: knee-only & brain-only (no combined)
# ----------------------------------------------------------

knee_path = [Path("/home/sq225/trial-project/data/knee-val/multicoil-val")]
brain_path = [Path("/home/sq225/trial-project/data/brain-val/multicoil-val")]

knee_data_module = FastMriDataModule(
    data_path=Path("fake bro"),
    challenge="multicoil",
    train_transform=custom_transform_combine_val,
    val_transform=custom_transform_combine_val,
    test_transform=custom_transform_combine_val,
    combine_diff_organs=True,
    batch_size=1,
    num_workers=1,
    data_paths_for_combine=knee_path,
    use_dataset_cache_file=False
)
brain_data_module = FastMriDataModule(
    data_path=Path("fake bro"),
    challenge="multicoil",
    train_transform=custom_transform_combine_val,
    val_transform=custom_transform_combine_val,
    test_transform=custom_transform_combine_val,
    combine_diff_organs=True,
    batch_size=1,
    num_workers=1,
    data_paths_for_combine=brain_path,
    use_dataset_cache_file=False
)

knee_loader = knee_data_module.val_dataloader()
brain_loader = brain_data_module.val_dataloader()

# ----------------------------------------------------------
# 2) Load “vanilla” (1-cascade) models with user snippet
#    "random vanilla" and "latent vanilla" are basically
#    the same architecture, but differ in how we feed the
#    latent vector (random vs. encoder).
# ----------------------------------------------------------

def load_trained_model(checkpoint_path, device="cuda"):
    """
    Loads a 1-cascade LatentVarNet from your own training checkpoint.
    """
    model = MyVarNet(num_cascades=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # dummy
    model, _, _ = load_checkpoint(model, optimizer, checkpoint_path)
    model.to(device)
    model.eval()
    return model

random_vanilla_ckpt = "/home/sq225/trial-project/models/varnet-random-latent-vector-attempt/checkpoints/varnet-random-latent-vector-attempt_best.pth"
latent_vanilla_ckpt = "/home/sq225/trial-project/models/varnet-latent-vector-attempt-100ep/checkpoints/varnet-latent-vector-attempt-100ep_best.pth"

model_random_vanilla = load_trained_model(random_vanilla_ckpt)
model_latent_vanilla = load_trained_model(latent_vanilla_ckpt)

# We'll also assume we have a pretrained encoder (resnet) loaded as `encoder`
# for "latent vector from pretrained" usage. 
# e.g.:
# encoder = load_resnet_encoder(...)
# encoder.eval()

# ----------------------------------------------------------
# 3) Load “pretrained” (12-cascade) baseline
# ----------------------------------------------------------


def load_pretrained_varnet(varnet_model_path, device="cuda"):
    """
    Load the official 12-cascade knee or brain pretrained baseline.
    """
    pretrained_varnet = OfficialVarNet(
        num_cascades=12,
        sens_chans=8,
        sens_pools=4,
        chans=18,
        pools=4,
        mask_center=True,
    )
    pretrained_varnet.load_state_dict(torch.load(varnet_model_path, map_location=device))
    pretrained_varnet.to(device)
    pretrained_varnet.eval()
    return pretrained_varnet

model_knee_pretrained = load_pretrained_varnet("knee_leaderboard_state_dict.pt")
model_brain_pretrained = load_pretrained_varnet("brain_leaderboard_state_dict.pt")

# ----------------------------------------------------------
# 4) Optionally do "latent knee-pretrained" or "latent brain-pretrained"
#    if the weights match your latent-based architecture.
#    We'll just assume we can load them into a "LatentVarNet(num_cascades=12)"
#    and that they line up.
# ----------------------------------------------------------

# def load_latent_pretrained(knee_or_brain_ckpt, device="cuda"):
#     # e.g. a 12-cascade LatentVarNet
#     try:
#       model = MyVarNet(num_cascades=12)  # must match the # of cascades
#       state = torch.load(knee_or_brain_ckpt, map_location=device)
#       model.load_state_dict(state, strict=True)
#       model.to(device)
#       model.eval()
#       return model
#     except Exception as e:
#         print(f"Can't load pretrained state_dict into MyVarNet: {e}")

# latent_knee_pretrained = load_latent_pretrained("knee_leaderboard_state_dict.pt")
# latent_brain_pretrained = load_latent_pretrained("brain_leaderboard_state_dict.pt")

# ----------------------------------------------------------
# 5) Evaluate all 6 configurations on knee & brain datasets
#    We'll define a function to compute PSNR & SSIM
# ----------------------------------------------------------
ssim = evaluate.ssim
psnr = evaluate.psnr

def evaluate_model(model, loader, device, use_random_vector=False, has_latent_param = False, encoder=None):
    """
    If use_random_vector=True, feed random vectors to VarNet.
    If use_latent_vector=True, feed encoder-latent from quick_recon_rss.
    Otherwise feed zero vector or do nothing if the model doesn't use it.
    """
    psnr_vals, ssim_vals = [], []

    model.eval()
    for batch in loader:
        # each batch from your custom transforms: 
        # masked_kspace, mask, quick_recon_rss, target, fname, slice_num, sens_maps
        masked_kspace, mask, quick_recon, target, fnames, slice_nums, sens_maps = batch
        masked_kspace = masked_kspace.to(device)
        mask = mask.to(device)
        quick_recon = quick_recon.to(device)
        target = target.to(device)
        sens_maps = sens_maps.to(device)
        
        # Decide how to get latent_vector 
        if encoder is not None:
          with torch.no_grad():
            latent_vector, _ = encoder(quick_recon)

        if use_random_vector:
            # e.g. if NormUnet was configured with latent_dim=128
            latent_vector = torch.randn_like(latent_vector)

        with torch.no_grad():
            # some models (the original 12-cascade) ignore the last 2 arguments 
            if has_latent_param:
                output = model(masked_kspace, mask, latent_vector, sens_maps)
            else:
                # masked_kspace = torch.nan_to_num(masked_kspace, nan=0.0, posinf=1.0, neginf=-1.0)  # Fix NaNs
                # masked_kspace = torch.clamp(masked_kspace, min=-1e3, max=1e3)  # Prevent extreme values
                # masked_kspace = masked_kspace / (masked_kspace.abs().max() + 1e-6)  # Normalize to a safe range

                masked_kspace = masked_kspace + 1e-8  # Add epsilon to prevent division by zero

                output = model(masked_kspace, mask)

        # center-crop output & target
        output, target = transforms.center_crop_to_smallest(output, target)

        # convert to numpy
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        # print("Output shape:", output.shape)
        # print("Target shape:", target.shape)
        
        # compute psnr, ssim
        psnr_val = psnr(output, target)
        ssim_val = ssim(output, target)
        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)

    return np.mean(psnr_vals), np.mean(ssim_vals)

# ----------------------------------------------------------
# Evaluate all 6:
#  1) random vanilla
#  2) knee-pretrained
#  3) brain-pretrained
#  4) latent vanilla (with an encoder)
#  5) latent knee-pretrained
#  6) latent brain-pretrained
# ----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = load_model("/home/sq225/trial-project/models/wandb-1st-attempt/checkpoints/checkpoint_9.pth")
encoder.to(device)
encoder.eval()

results = {}

def run_eval(model, loader_knee, loader_brain, tag, **kwargs):
    psnr_knee, ssim_knee = evaluate_model(model, loader_knee, device, **kwargs)
    psnr_brain, ssim_brain = evaluate_model(model, loader_brain, device, **kwargs)
    results[tag] = {
        "knee_psnr": psnr_knee,
        "knee_ssim": ssim_knee,
        "brain_psnr": psnr_brain,
        "brain_ssim": ssim_brain,
    }


print("Evaluating random vanilla")

# 1) random vanilla
run_eval(model_random_vanilla, knee_loader, brain_loader, "random_vanilla", use_random_vector=True, has_latent_param = True, encoder=encoder)

print("Evaluating knee-pretrained")
# 2) knee-pretrained (12-cascade, no latents)
run_eval(model_knee_pretrained, knee_loader, brain_loader, "knee_pretrained", use_random_vector=False, has_latent_param = False, encoder=None)

print("Evaluating brain-pretrained")
# 3) brain-pretrained (12-cascade, no latents)
run_eval(model_brain_pretrained, knee_loader, brain_loader, "brain_pretrained", use_random_vector=False, has_latent_param =False, encoder=None)

print("Evaluating latent vanilla")
# 4) latent vanilla (1-cascade, feed encoder vector)
run_eval(model_latent_vanilla, knee_loader, brain_loader, "latent_vanilla", use_random_vector=False, has_latent_param = True, encoder=encoder)

# # 5) latent knee-pretrained
# latent_knee = load_latent_pretrained("knee_leaderboard_state_dict.pt")
# run_eval(latent_knee, knee_loader, brain_loader, "latent_knee_pretrained", use_random_vector=False, use_latent_vector=True, encoder=encoder)

# # 6) latent brain-pretrained
# latent_brain = load_latent_pretrained("brain_leaderboard_state_dict.pt")
# run_eval(latent_brain, knee_loader, brain_loader, "latent_brain_pretrained", use_random_vector=False, use_latent_vector=True, encoder=encoder)

# ----------------------------------------------------------
# Print or plot results
# ----------------------------------------------------------
import pprint
pprint.pprint(results)

# Optionally, you can do a 2D scatter plot: 
#   X-axis = PSNR on knee, Y-axis = PSNR on brain
#   or do the same for SSIM.

import matplotlib.pyplot as plt

plt.figure()
for model_tag, vals in results.items():
    print("model tag", model_tag)
    plt.scatter(vals["knee_psnr"], vals["brain_psnr"], label=model_tag, s=60)
plt.xlabel("Knee PSNR (dB)")
plt.ylabel("Brain PSNR (dB)")
plt.legend()
plt.title("Knee vs Brain PSNR Comparison")
plt.show()
# save the figure
plt.savefig("knee_vs_brain_psnr_comparison_benchmark.png")

plt.figure()
for model_tag, vals in results.items():
    plt.scatter(vals["knee_ssim"], vals["brain_ssim"], label=model_tag, s=60)
plt.xlabel("Knee SSIM")
plt.ylabel("Brain SSIM")
plt.legend()
plt.title("Knee vs Brain SSIM Comparison")
# save the figure
plt.savefig("knee_vs_brain_ssim_comparison_benchmark.png")
