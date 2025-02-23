# File: save_sens_maps.py
import torch
from pathlib import Path
from datasetv2 import FastMriDataModule, custom_transform_combine
from fastmri.models.varnet import VarNet, NormUnet
import fastmri


class DebugNormUnet(NormUnet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        
        print("DebugNormUnet - Input min/max:", x.min().item(), x.max().item())
        if torch.isnan(x).any():
            print("ERROR: Input to NormUnet contains NaNs!")
            exit(1)

        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        print("After norm min/max:", x.min().item(), x.max().item())
        if torch.isnan(x).any():
            print("ERROR: NaNs after normalization in NormUnet!")
            exit(1)

        x, pad_sizes = self.pad(x)
        print("After padding min/max:", x.min().item(), x.max().item())

        x = self.unet(x)
        print("After U-Net min/max:", x.min().item(), x.max().item())
        if torch.isnan(x).any():
            print("ERROR: NaNs after U-Net layers in NormUnet!")
            exit(1)

        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        print("Final output of NormUnet min/max:", x.min().item(), x.max().item())
        if torch.isnan(x).any():
            print("ERROR: Final output of NormUnet contains NaNs!")
            exit(1)

        return x

def save_sensitivity_maps(varnet_model_path, save_dir):

    # create save_dir
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained VarNet
    pretrained_varnet = VarNet(
        num_cascades=12,
        sens_chans=8,
        sens_pools=4,
        chans=18,
        pools=4,
        mask_center=True,
    )
    pretrained_varnet.load_state_dict(torch.load(varnet_model_path))
    pretrained_varnet.eval()

    # Initialize DataModule (use same config as training)
    data_paths_train = [
        Path("/home/sq225/trial-project/data/brain-train/multicoil-train"),
        Path("/home/sq225/trial-project/data/knee-train/multicoil-train")
    ]

    data_paths_val = [
        Path("/home/sq225/trial-project/data/brain-val/multicoil-val"),
        Path("/home/sq225/trial-project/data/knee-val/multicoil-val")
    ]

    data_module_train = FastMriDataModule(
            data_path=Path("fake bro"),
            challenge="multicoil",
            train_transform=custom_transform_combine,
            val_transform=custom_transform_combine,
            test_transform=custom_transform_combine,
            batch_size=1,
            num_workers=1,
            combine_diff_organs=True,
            data_paths_for_combine=data_paths_train
        )

    data_module_val = FastMriDataModule(
            data_path=Path("fake bro"),
            challenge="multicoil",
            train_transform=custom_transform_combine,
            val_transform=custom_transform_combine,
            test_transform=custom_transform_combine,
            batch_size=1,
            num_workers=1,
            combine_diff_organs=True,
            data_paths_for_combine=data_paths_val
        )

    train_loader = data_module_train.train_dataloader()
    val_loader = data_module_val.val_dataloader()

    # Iterate through dataset and save sensitivity maps
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        split_dir = save_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for batch in loader:
            masked_kspace, mask, _, target, fname, slice_num = batch

            print("masked_kspace min/max before sens_net:", masked_kspace.min().item(), masked_kspace.max().item())
            if torch.isnan(masked_kspace).any():
                print("ERROR: `masked_kspace` contains NaNs before `sens_net`!")
                exit(1)

            print("mask min/max before sens_net:", mask.min().item(), mask.max().item())
            if torch.isnan(mask).any():
                print("ERROR: `mask` contains NaNs before `sens_net`!")
                exit(1)


            masked_kspace = torch.nan_to_num(masked_kspace, nan=0.0, posinf=1.0, neginf=-1.0)  # Fix NaNs
            masked_kspace = torch.clamp(masked_kspace, min=-1e3, max=1e3)  # Prevent extreme values
            masked_kspace = masked_kspace / (masked_kspace.abs().max() + 1e-6)  # Normalize to a safe range

            masked_kspace = masked_kspace + 1e-8  # Add epsilon to prevent division by zero


            # these are for checking the operations in the sens_net
            images = fastmri.ifft2c(masked_kspace)
            print("images min/max in ifft2c in sens_net:", images.min().item(), images.max().item())
            if torch.isnan(images).any():
                print("ERROR: `images` contains NaNs in `sens_net`!")
                exit(1)
            b, c, h, w, comp = images.shape
            images = images.view(b * c, 1, h, w, comp)
            print("images after chans_to_batch_dim in sens_net:", images.min().item(), images.max().item())
            if torch.isnan(images).any():
                print("ERROR: `images` contains NaNs in `sens_net`!")
                exit(1)
            # override with debugging unet 
            pretrained_varnet.sens_net.norm_unet = DebugNormUnet(chans=8, num_pools=4)




            with torch.no_grad():
                sens_maps = pretrained_varnet.sens_net(masked_kspace, mask)

            # Save sens_maps with fname and slice_num as identifiers
            for idx in range(len(fname)):
                
                fname_stem = Path(fname[idx]).stem  # Removes extension
                fname_dir = split_dir / fname_stem
                fname_dir.mkdir(parents=True, exist_ok=True)

                # Check sens_maps for nans before saving
                if torch.isnan(sens_maps).any():
                    print(f"ERROR: `sens_maps` contains NaNs before saving! Filename: {fname_stem}, Slice: {slice_num}")
                    exit(1)

                torch.save(
                    sens_maps[idx],
                    fname_dir / f"sens_map_slice{slice_num[idx]}.pt",
                )
                
if __name__ == "__main__":
    save_sensitivity_maps(varnet_model_path="knee_leaderboard_state_dict.pt", save_dir="sens_maps_after_checking_fft") # for testing, usually use sens_maps2