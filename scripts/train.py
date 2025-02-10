import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import MRIClassifier
from dataset import FastMRIClassificationDataset

import wandb
import utils
import os

def train():

    MODEL_NAME = "wandb-1st-attempt-on-mac"
    MODELS_DIR = '~/trial-project/models/'
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_dir = os.path.join(MODELS_DIR, MODEL_NAME)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    # 1. Create dataset objects
    train_dataset = FastMRIClassificationDataset(root_dir="~/trial-project/data/", split='train', center_crop_size=(128, 128))
    val_dataset = FastMRIClassificationDataset(root_dir="~/trial-project/data/", split='val', center_crop_size=(128, 128))
    
    # 2. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
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
        "epochs": 10,
        }
    )

    # Track the best validation accuracy for checkpointing
    best_val_acc = 0.0

    # 5. Training loop (simplified)
    for epoch in range(1, 11):
        model.train()
        for batch in train_loader:
            images, labels = batch  # images = [B, C, H, W], labels = [B]
        
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            latent_vector, logits = model(images)
            
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
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    _, logits = model(images)
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
                utils.save_model(f"{MODEL_NAME}_best.pth", model_dir, model.state_dict())
        

if __name__ == '__main__':
    train()