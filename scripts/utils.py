import os
import torch
from model import MRIClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def extract_latent(model, image_tensor):
    """
    image_tensor: [C, H, W] or [1, C, H, W]
    returns: latent_vector of size [embed_dim]
    """
    model.eval()
    with torch.no_grad():
        # If single image, unsqueeze to create batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        latent_vector, _ = model(image_tensor)
    return latent_vector

def load_model(checkpoint_dir, map_location='cpu'):
  model = MRIClassifier(num_classes=2, embed_dim=512, pretrained=False)
  checkpoint = torch.load(checkpoint_dir,map_location=map_location)
  model.load_state_dict(checkpoint['model_state_dict'])
  print(f"Loaded model from epoch {checkpoint['epoch']}")

  return model

def save_model(model_name, model_dir, model_state_dict):
  """
  Save the model state dictionary to a file.

  Args:
    model_name (str): The name of the model.
    model_dir (str): The directory where the model will be saved.
    model_state_dict (dict): The state dictionary of the model.

  Returns:
    None
  """
  model_save_path = os.path.join(model_dir, model_name)
  os.makedirs(model_dir, exist_ok=True)
  torch.save(model_state_dict, f=model_save_path)
  print(f"Model saved at {model_save_path}")

def save_checkpoint(model, optimizer, epoch, dir='checkpoints', filename='checkpoint.pth'):
  """
  Save the model, optimizer, and epoch information to a checkpoint file.

  Args:
    model (torch.nn.Module): The model to be saved.
    optimizer (torch.optim.Optimizer): The optimizer to be saved.
    epoch (int): The current epoch.
    dir (str): The directory where the checkpoint will be saved. Default is 'checkpoints'.
    filename (str): The filename of the checkpoint file. Default is 'checkpoint.pth'.

  Returns:
    None
  """
  os.makedirs(dir, exist_ok=True)
  save_path = os.path.join(dir, filename)
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
  }, save_path)
  print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, checkpoint_path='checkpoint.pth'):
  """
  Load the model, optimizer, and epoch information from a checkpoint file.

  Args:
    model (torch.nn.Module): The model to be loaded.
    optimizer (torch.optim.Optimizer): The optimizer to be loaded.
    checkpoint_path (str): The path to the checkpoint file. Default is 'checkpoint.pth'.

  Returns:
    model (torch.nn.Module): The loaded model.
    optimizer (torch.optim.Optimizer): The loaded optimizer.
    epoch (int): The epoch loaded from the checkpoint.
  """
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  print(f"Checkpoint loaded at epoch {epoch}")
  return model, optimizer, epoch