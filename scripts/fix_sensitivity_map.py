# File: save_sens_maps.py
import torch
from pathlib import Path
from datasetv2 import FastMriDataModule, custom_transform_combine
from fastmri.models.varnet import VarNet

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

            with torch.no_grad():
                sens_maps = pretrained_varnet.sens_net(masked_kspace, mask)

            # Save sens_maps with fname and slice_num as identifiers
            for idx in range(len(fname)):
                
                fname_stem = Path(fname[idx]).stem  # Removes extension
                fname_dir = split_dir / fname_stem
                fname_dir.mkdir(parents=True, exist_ok=True)

                torch.save(
                    sens_maps[idx],
                    fname_dir / f"sens_map_slice{slice_num[idx]}.pt",
                )
                
if __name__ == "__main__":
    save_sensitivity_maps(varnet_model_path="knee_leaderboard_state_dict.pt", save_dir="sens_maps2")