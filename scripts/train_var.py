import torch
import torch.optim as optim
import wandb
from fastmri.losses import SSIMLoss
from datasetv2 import custom_transform_combine, FastMriDataModule
from varnet_module_v3 import LatentVarNet
from pathlib import Path
from utils import load_model, save_checkpoint
from varnet_module_v3 import download_varnet_model, save_sensitivity_maps
import os

MODEL_NAME = "varnet-1st-attempt-on-mac"

# MODELS_DIR = '/home/sq225/trial-project/models/'
MODELS_DIR = '/Users/ericq/trial-project/models/'

os.makedirs(MODELS_DIR, exist_ok=True)
model_dir = os.path.join(MODELS_DIR, MODEL_NAME)
checkpoint_dir = os.path.join(model_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Hyperparameters
epochs = 50
batch_size = 4
learning_rate = 1e-4
latent_dim = 128

# Define URLs and paths
varnet_model_url = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/knee_leaderboard_state_dict.pt"
varnet_model_path = "knee_leaderboard_state_dict.pt"
sens_maps_path = "fixed_sens_maps.pth"

# Download and extract sensitivity maps
if not os.path.exists(varnet_model_path):
    download_varnet_model(varnet_model_url, varnet_model_path)
save_sensitivity_maps(varnet_model_path, sens_maps_path)

# DataModule Setup


data_paths_train = [
        Path("/Users/ericq/trial-project/data/brain-train/multicoil-train"),
        Path("/Users/ericq/trial-project/data/knee-train/multicoil-train"),
    ]

data_paths_val = [
        Path("/Users/ericq/trial-project/data/brain-val/multicoil-val"),
        Path("/Users/ericq/trial-project/data/knee-val/multicoil-val"),
    ]

# data_paths_train = [
#     Path("/home/sq225/trial-project/data/brain-train/multicoil-train"),
#     Path("/home/sq225/trial-project/data/knee-train/multicoil-train")
# ]

# data_paths_val = [
#     Path("/home/sq225/trial-project/data/brain-val/multicoil-val"),
#     Path("/home/sq225/trial-project/data/knee-val/multicoil-val")
# ]

data_module_train = FastMriDataModule(
        data_path=Path("fake bro"),
        challenge="multicoil",
        train_transform=custom_transform_combine,
        val_transform=custom_transform_combine,
        test_transform=custom_transform_combine,
        batch_size=4,
        num_workers=2,
        combine_diff_organs=True,
        data_paths_for_combine=data_paths_train
    )

data_module_val = FastMriDataModule(
        data_path=Path("fake bro"),
        challenge="multicoil",
        train_transform=custom_transform_combine,
        val_transform=custom_transform_combine,
        test_transform=custom_transform_combine,
        batch_size=4,
        num_workers=2,
        combine_diff_organs=True,
        data_paths_for_combine=data_paths_val
    )

train_loader = data_module_train.train_dataloader()
val_loader = data_module_val.val_dataloader()


# Models
varnet = LatentVarNet(sense_maps_path="fixed_sens_maps.pth", latent_dim=latent_dim)

# Encoder
# encoder = load_model("/home/sq225/trial-project/models/wandb-1st-attempt/checkpoints/checkpoint_9.pth")
encoder = load_model("/Users/ericq/trial-project/models/wandb-1st-attempt/checkpoints/checkpoint_9.pth")

# Loss and Optimizer
criterion = SSIMLoss()
optimizer = optim.Adam(varnet.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
varnet.to(device)
encoder.to(device)
encoder.eval()

# Track the best validation loss for checkpointing
best_val_loss = float("inf")

# Initialize Weights & Biases (Wandb)
wandb.init(project="latent-varnet-training", name=MODEL_NAME, config={
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "latent_dim": latent_dim
})

# Training Loop
for epoch in range(epochs):
    varnet.train()
    running_loss = 0.0

    for batch in train_loader:
        masked_kspace, mask, quick_recon_rss, target, fname, slice_num = batch
        masked_kspace, mask, quick_recon_rss, target = (
            masked_kspace.to(device),
            mask.to(device),
            quick_recon_rss.to(device),
            target.to(device),
        )

        # Step 1: Get latent vector from the pre-trained encoder
        with torch.no_grad():
            latent_vector, _ = encoder(quick_recon_rss)

        # Step 2: Pass k-space and latent vector to VarNet
        output = varnet(masked_kspace, mask, latent_vector)

        # Compute loss
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        wandb.log({"train_loss": loss.item()})

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

    # Validation Loop
    varnet.eval()
    val_loss = 0.0
    if epoch % 5 == 0:
        with torch.no_grad():
            for batch in val_loader:
                masked_kspace, mask, quick_recon_rss, target, fname, slice_num = batch
                masked_kspace, mask, quick_recon_rss, target = (
                    masked_kspace.to(device),
                    mask.to(device),
                    quick_recon_rss.to(device),
                    target.to(device),
                )

                latent_vector, _ = encoder(quick_recon_rss)
                output = varnet(masked_kspace, mask, latent_vector)
                loss = criterion(output, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch}/{epochs}], Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss})

        # Save Checkpoints
        save_checkpoint(varnet, optimizer, epoch, dir=checkpoint_dir, filename=f"checkpoint_{epoch}.pth")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(varnet, optimizer, epoch, dir=checkpoint_dir, filename=f"{MODEL_NAME}_best.pth")
            print("Saved best model checkpoint.")

# Finalize Wandb
wandb.finish()
print("Training completed successfully!")