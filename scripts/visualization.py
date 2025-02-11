import torch
from torch.utils.data import DataLoader
from dataset import FastMRIClassificationDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import load_model

def collect_latent_vectors(model, dataloader, device='cpu'):
    """
    Given a model and a dataset, return a list/array of:
        - latent_vectors: shape [N, embed_dim]
        - labels: shape [N]
    where N = len(dataset).
    """
    
    all_latent_vectors = []
    all_labels = []
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for images, labels in dataloader:

            # track progress
            print(f"Processing batch {len(all_labels)}")

            images = images.to(device)
            # forward pass
            latent_vector, _ = model(images)  # we only need the latent part
            # move to CPU for easy handling (e.g., for NumPy or scikit-learn)
            latent_vector = latent_vector.cpu()
            labels = labels.cpu()

            all_latent_vectors.append(latent_vector)
            all_labels.append(labels)
    
    # Concatenate all batches
    all_latent_vectors = torch.cat(all_latent_vectors, dim=0)  # shape [N, embed_dim]
    all_labels = torch.cat(all_labels, dim=0)  # shape [N]

    return all_latent_vectors, all_labels

def visualize_tsne(latent_vectors, labels, perplexity=30):
    """
    Perform t-SNE on the latent vectors and make a scatter plot.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    
    # latent_vectors is a torch.Tensor of shape [N, embed_dim]; convert to numpy
    latent_np = latent_vectors.numpy()
    
    latent_2d = tsne.fit_transform(latent_np)  # shape [N, 2]
    
    plt.figure(figsize=(8,6))
    
    # Convert labels to numpy, too
    labels_np = labels.numpy()
    
    # Plot each point, colored by label (0 for brain, 1 for knee, or vice versa)
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels_np, cmap='viridis')
    plt.colorbar(scatter, ticks=[0,1], label='Organ label')
    plt.title("t-SNE of Latent Vectors (Brain vs. Knee)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("tsne_plot.png", dpi=300)

# from model import MRIClassifier  # or however you import it
# from dataset import FastMRIClassificationDataset

def main():

    model = load_model("/home/sq225/trial-project/models/wandb-1st-attempt/checkpoints/checkpoint_9.pth")

    print("starting creating dataset")

    # 2. Create a dataset for validation or test
    val_dataset = FastMRIClassificationDataset(
        root_dir="/home/sq225/trial-project/data/", 
        split='val',
        center_crop_size=(128, 128)
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print("starting collecting latent vectors")

    # 3. Extract latent vectors
    latent_vectors, labels = collect_latent_vectors(model, val_loader, device='cuda')

    print("starting visualizing")

    # 4. Visualize with t-SNE or UMAP
    visualize_tsne(latent_vectors, labels)
    # Or:
    # visualize_umap(latent_vectors, labels)

if __name__ == "__main__":
    main()