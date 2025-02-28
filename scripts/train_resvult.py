import torch
import torch.optim as optim
import wandb
from datasetvult import custom_transform_combine_train, custom_transform_combine_val, FastMriDataModule
from pathlib import Path
from model import MRIClassifier
from utils import load_model, save_checkpoint
from fastmri.data import transforms
import torch.nn.functional as F
# from varnet_module_vds import download_varnet_model, save_sensitivity_maps
import os
from torch import nn
import utils



def train():

    MODEL_NAME = "resnet18--100ep-128dim-ifft-crop-fft"
    MODELS_DIR = '/home/sq225/trial-project/resnet-models/'
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_dir = os.path.join(MODELS_DIR, MODEL_NAME)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device selection
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
            batch_size=1,
            num_workers=1,
            combine_diff_organs=True,
            data_paths_for_combine=data_paths_train,
            use_dataset_cache_file=False
        )

    data_module_val = FastMriDataModule(
            data_path=Path("fake bro"),
            challenge="multicoil",
            train_transform=custom_transform_combine_train,
            val_transform=custom_transform_combine_val,
            test_transform=custom_transform_combine_val,
            batch_size=1,
            num_workers=1,
            combine_diff_organs=True,
            data_paths_for_combine=data_paths_val,
            use_dataset_cache_file=False
        )

    train_loader = data_module_train.train_dataloader()
    val_loader = data_module_val.val_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Instantiate the model
    model = MRIClassifier(num_classes=2, embed_dim=128, pretrained=True)
    model = model.to(device)
    
    # 4. Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    wandb.init(
        # set the wandb project where this run will be logged
        project="trial-project",

        name=MODEL_NAME,

        # track hyperparameters and run metadata
        config={
        "learning_rate": 1e-4,
        "architecture": "resnet18",
        "dataset": "fastMRI",
        "epochs": 100,
        }
    )

    # Track the best validation accuracy for checkpointing
    best_val_acc = 0.0

    # 5. Training loop (simplified)
    for epoch in range(1, 101):
        model.train()
        for batch in train_loader:
            masked_kspace, mask, quick_recon_rss, target, fname, slice_num, sens_maps = batch
            masked_kspace, mask, quick_recon_rss, target, sens_maps = (
                masked_kspace.to(device),
                mask.to(device),
                quick_recon_rss.to(device),
                target.to(device),
                sens_maps.to(device)
            )
            
            # Forward pass
            latent_vector, logits = model(quick_recon_rss)

            labels = torch.tensor([1 if "brain" in f else 0 for f in fname]).to(device)
            
            # Compute classification loss
            loss = loss_fn(logits, labels)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation step (illustrative)
        model.eval()
        correct = 0
        total = 0
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

                    labels = torch.tensor([1 if "brain" in f else 0 for f in fname]).to(device)
                    
                    _, logits = model(quick_recon_rss)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            
            val_acc = 100.0 * correct / total
            print(f"Epoch {epoch}, Val Accuracy: {val_acc:.2f}%")
            wandb.log({"val_accuracy": val_acc})
            utils.save_checkpoint(model, optimizer, epoch, dir=checkpoint_dir, filename=f"checkpoint_{epoch}.pth")

            # Save the model if it has the best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                utils.save_checkpoint(model, optimizer, epoch, dir=checkpoint_dir, filename=f"{MODEL_NAME}_best.pth")
                print("Saved best model checkpoint.")
        

if __name__ == '__main__':
    train()