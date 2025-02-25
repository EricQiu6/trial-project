import torch
import matplotlib.pyplot as plt
from train_var_from_g2 import train
from datasetv2 import custom_transform_combine_train, custom_transform_combine_val, FastMriDataModule
from utils import load_checkpoint, load_model
from fastmri.data import transforms
from pathlib import Path
from varnet_module_vds import LatentVarNet

# Load trained model checkpoint
def load_trained_model(checkpoint_path, device="cuda"):
    model = LatentVarNet(num_cascades=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Dummy optimizer
    model, _, _ = load_checkpoint(model, optimizer, checkpoint_path)
    model.to(device)
    model.eval()
    return model

# Visualize a sample batch
def visualize_reconstructions(model, dataloader, device="cuda", save_path=None):
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        masked_kspace, mask, quick_recon_rss, target, fname, slice_num, sens_maps = batch
        
        masked_kspace, mask, quick_recon_rss, target, sens_maps = (
            masked_kspace.to(device), mask.to(device), quick_recon_rss.to(device),
            target.to(device), sens_maps.to(device)
        )
        
        # Extract latent vector from encoder
        encoder = load_model("/home/sq225/trial-project/resnet-models/resnet18-varnet-loader-100ep/checkpoints/resnet18-varnet-loader-100ep_best.pth")
        encoder.to(device)
        encoder.eval()
        latent_vector, _ = encoder(quick_recon_rss)

        # TEST RANDOM LATENT VECTOR - verified doesn't work
        # latent_vector = torch.randn_like(latent_vector)
        
        # Perform reconstruction
        output = model(masked_kspace=masked_kspace, mask=mask, latent_vector=latent_vector, sens_maps=sens_maps)
        
        print("Masked k-space min/max:", masked_kspace.min().item(), masked_kspace.max().item())
        print("Latent vector min/max:", latent_vector.min().item(), latent_vector.max().item())
        print("Model output min/max before cropping:", output.min().item(), output.max().item())



        # Ensure correct shapes
        output, target = transforms.center_crop_to_smallest(output, target)
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Convert to numpy for visualization
        output_np = output.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(target_np, cmap='gray')
        axes[0].set_title("Ground Truth")
        
        axes[1].imshow(output_np, cmap='gray')
        axes[1].set_title("Reconstructed")
        
        plt.suptitle(f"Reconstruction vs. Ground Truth\nFile: {fname[0]} - Slice: {slice_num[0]}")
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    checkpoint_path = "/home/sq225/trial-project/models/varnet-latent-vector-no-coil-no-latent-crop/checkpoints/varnet-latent-vector-no-coil-no-latent-crop_best.pth"  # Update with actual checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = load_trained_model(checkpoint_path, device)
    
    # Load dataset
    data_paths_val = [
        Path("/home/sq225/trial-project/data/brain-val/multicoil-val"),
        Path("/home/sq225/trial-project/data/knee-val/multicoil-val")
    ]

    data_module_val = FastMriDataModule(
            data_path="fake bro",
            challenge="multicoil",
            train_transform=custom_transform_combine_train,
            val_transform=custom_transform_combine_val,
            test_transform=custom_transform_combine_val,
            batch_size=1,
            num_workers=1,
            combine_diff_organs=True,
            data_paths_for_combine=data_paths_val
        )
    val_loader = data_module_val.val_dataloader()
    
    # Run visualization
    visualize_reconstructions(model, val_loader, device, "/home/sq225/trial-project/scripts/visualization_varnet-latent-vector-no-coil-no-latent-crop_best.pth.png")  # Update with actual save path
