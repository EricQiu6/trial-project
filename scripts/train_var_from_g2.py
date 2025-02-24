import torch
import torch.optim as optim
import wandb
from fastmri.losses import SSIMLoss
from datasetv2 import custom_transform_combine_train, custom_transform_combine_val, FastMriDataModule
from varnet_module_vds import LatentVarNet
from pathlib import Path
from utils import load_model, save_checkpoint
from fastmri.data import transforms
import torch.nn.functional as F
# from varnet_module_vds import download_varnet_model, save_sensitivity_maps
import os

def train():

    MODEL_NAME = "varnet-1st-attempt-after-sens-map-fix"
    MODELS_DIR = '/home/sq225/trial-project/models/'
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_dir = os.path.join(MODELS_DIR, MODEL_NAME)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Hyperparameters
    epochs = 50
    batch_size = 1
    learning_rate = 1e-4
    latent_dim = 128

    # Define URLs and paths
    varnet_model_url = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/knee_leaderboard_state_dict.pt"
    varnet_model_path = "knee_leaderboard_state_dict.pt"
    sens_maps_path = "fixed_sens_maps.pth"

    # DataModule Setup


    # data_paths_train = [
    #         Path("/Users/ericq/trial-project/official-fitting-data/brain/multicoil_train"),
    #         Path("/Users/ericq/trial-project/official-fitting-data/knee/multicoil_train"),
    #     ]

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
            train_transform=custom_transform_combine_train,
            val_transform=custom_transform_combine_val,
            test_transform=custom_transform_combine_val,
            batch_size=batch_size,
            num_workers=1,
            combine_diff_organs=True,
            data_paths_for_combine=data_paths_train
        )

    data_module_val = FastMriDataModule(
            data_path=Path("fake bro"),
            challenge="multicoil",
            train_transform=custom_transform_combine_train,
            val_transform=custom_transform_combine_val,
            test_transform=custom_transform_combine_val,
            batch_size=batch_size,
            num_workers=1,
            combine_diff_organs=True,
            data_paths_for_combine=data_paths_val
        )

    train_loader = data_module_train.train_dataloader()
    val_loader = data_module_val.val_dataloader()


    # Models
    varnet = LatentVarNet(num_cascades=1)

    # Encoder
    encoder = load_model("/home/sq225/trial-project/models/wandb-1st-attempt/checkpoints/checkpoint_9.pth")
    # encoder = load_model("/Users/ericq/trial-project/models/wandb-1st-attempt/checkpoints/checkpoint_9.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loss and Optimizer
    criterion = SSIMLoss().to(device)
    # I don't think adding weight decay helps
    optimizer = optim.Adam(varnet.parameters(), lr=learning_rate)
    varnet.to(device)
    encoder.to(device)
    encoder.eval()

    # Track the best validation loss for checkpointing
    best_val_loss = float("inf")

    # Initialize Weights & Biases (Wandb)
    # wandb.init(project="latent-varnet-training", name=MODEL_NAME, config={
    #     "epochs": epochs,
    #     "batch_size": batch_size,
    #     "learning_rate": learning_rate,
    #     "latent_dim": latent_dim
    # })

    # Training Loop
    for epoch in range(epochs):
        varnet.train()
        running_loss = 0.0

        for batch in train_loader:
            masked_kspace, mask, quick_recon_rss, target, fname, slice_num, sens_maps = batch
            masked_kspace, mask, quick_recon_rss, target, sens_maps = (
                masked_kspace.to(device),
                mask.to(device),
                quick_recon_rss.to(device),
                target.to(device),
                sens_maps.to(device)
            )

            # Step 1: Get latent vector from the pre-trained encoder
            with torch.no_grad():
                latent_vector, _ = encoder(quick_recon_rss)

            # Step 2: Pass k-space and latent vector to VarNet
            output = varnet(masked_kspace=masked_kspace, mask=mask, latent_vector=latent_vector, sens_maps=sens_maps)

            # for name, param in varnet.named_parameters():
            #     if param.grad is None:
            #         print(f"WARNING: No gradient for {name}")
            #     else:
            #         print(f"Gradient norm for {name}: {torch.norm(param.grad)}")

            # Ensure output and target have the same shape
            output, target = transforms.center_crop_to_smallest(output, target)

            # Ensure valid output and target
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)

            # Compute data_range safely
            data_range = target.max() - target.min()
            if data_range == 0 or torch.isnan(data_range):
                print("Warning: Invalid data_range, setting to 1.0")
                data_range = torch.tensor(1.0).to(device)
            
            if len(data_range.shape) == 0:
                data_range = data_range.unsqueeze(0)

            # Compute SSIM + L1 Loss
            ssim_loss = criterion(output, target, data_range)  # SSIMLoss already returns 1 - SSIM
            l1_loss = F.l1_loss(output, target)
            loss = ssim_loss + 0.1 * l1_loss  # Avoid NaN loss

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(varnet.parameters(), max_norm=1.0)

            optimizer.zero_grad()
            loss.backward()

            for name, param in varnet.named_parameters():
                if param.grad is None:
                    print(f"WARNING: No gradient for {name}")
                else:
                    # print(f"Gradient norm for {name}: {torch.norm(param.grad)}")
                    if torch.isnan(param.grad).any():
                        print(f"ERROR: NaN detected in gradient of {name}")
                        exit(1)

            optimizer.step()

            running_loss += loss.item()
            # wandb.log({"train_loss": loss.item()})

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation Loop
        varnet.eval()
        val_loss = 0.0
        if epoch % 3 == 0:
            with torch.no_grad():
                for batch in val_loader:
                    masked_kspace, mask, quick_recon_rss, target, fname, slice_num, sens_maps = batch
                    masked_kspace, mask, quick_recon_rss, target, sens_maps = (
                        masked_kspace.to(device),
                        mask.to(device),
                        quick_recon_rss.to(device),
                        target.to(device),
                        sens_maps.to(device)
                        
                    )

                    # why the fuck would the sens map be nan still wuwuuwuwuwuwu
                    if torch.isnan(sens_maps).any():
                        print("ERROR: `sens_map` loaded from batch contains NaNs")
                        exit(1)

                    latent_vector, _ = encoder(quick_recon_rss)
                    output = varnet(masked_kspace=masked_kspace, mask=mask, latent_vector=latent_vector, sens_maps=sens_maps)
                    
                    # Ensure output and target have the same shape
                    output, target = transforms.center_crop_to_smallest(output, target)
                    
                    # Ensure valid output and target
                    output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
                    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)

                    # Compute data_range safely
                    data_range = target.max() - target.min()
                    if data_range == 0 or torch.isnan(data_range):
                        print("Warning: Invalid data_range, setting to 1.0")
                        data_range = torch.tensor(1.0).to(device)
                    
                    if len(data_range.shape) == 0:
                        data_range = data_range.unsqueeze(0)
                        
                    loss = criterion(output, target, data_range)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch}/{epochs}], Validation Loss: {avg_val_loss:.4f}")
            # wandb.log({"val_loss": avg_val_loss})

            # Save Checkpoints
            save_checkpoint(varnet, optimizer, epoch, dir=checkpoint_dir, filename=f"checkpoint_{epoch}.pth")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(varnet, optimizer, epoch, dir=checkpoint_dir, filename=f"{MODEL_NAME}_best.pth")
                print("Saved best model checkpoint.")

    # Finalize Wandb
    # wandb.finish()
    print("Training completed successfully!")

if __name__ == '__main__':
    train()