import torch
import matplotlib.pyplot as plt
from pathlib import Path
from fastmri import fft2c, complex_abs
import sys
import random

def load_sensitivity_map(sens_map_path):
    print(f"Loading sensitivity map from: {sens_map_path}")
    sens_maps = torch.load(sens_map_path)
    print("Sensitivity map min/max:", sens_maps.min().item(), sens_maps.max().item())
    if torch.isnan(sens_maps).any():
        print("ERROR: `sens_maps` contains NaNs!")
        exit(1)
    return sens_maps

def visualize_sensitivity_map(sens_maps, fname, folder):
    num_coils = sens_maps.shape[0]
    print("Sensitivity map shape:", sens_maps.shape)
    print(f"Number of coils: {num_coils}")
    
    # Compute FFT of sensitivity maps
    fft_sens_maps = fft2c(sens_maps)
    print("FFT Sens. Map shape:", fft_sens_maps.shape)
    fft_sens_maps_abs = complex_abs(fft_sens_maps)
    print("FFT Sens. Map min/max:", fft_sens_maps_abs.min().item(), fft_sens_maps_abs.max().item())
    
    if torch.isnan(fft_sens_maps_abs).any():
        print("ERROR: FFT of `sens_maps` contains NaNs!")
        exit(1)
    
    # Visualize sensitivity maps and their FFTs
    plt.figure(figsize=(18, 6))
    for i in range(num_coils):
        plt.subplot(2, num_coils, i+1)

        # see magnitude of the sensitivity maps
        sens_maps_abs = complex_abs(sens_maps[i])
        print("sens_maps_abs shape:", sens_maps_abs.shape)

        plt.imshow(sens_maps_abs)
        plt.title(f'Sens. Map Coil {i}')
        plt.axis('off')
        
        plt.subplot(2, num_coils, num_coils + i+1)

        print("fft_sens_maps_abs shape:", fft_sens_maps_abs[i].shape)

        plt.imshow(torch.log1p(fft_sens_maps_abs[i]).cpu().numpy())
        plt.title(f'FFT Coil {i}')
        plt.axis('off')
    
    plt.suptitle(f"Sensitivity Maps and FFTs - {fname}")
    plt.tight_layout()
    plt.savefig(f"{fname}_sensitivity_maps_from_sens_map2.png")

def main():
    # Path to the saved sensitivity map
    # Expect the folder path as the first command line argument.
    
    if len(sys.argv) < 2:
      print("Usage: python fft_sens_map_visualization.py <folder_path>")
      exit(1)
    folder = Path(sys.argv[1])
    
    # Search for files matching the sensitivity map pattern recursively.
    sens_map_files = sorted(folder.glob("**/sens_map_slice*.pt"))
    if not sens_map_files:
      print(f"No sensitivity map files found in {folder}")
      exit(1)
    
    # For demonstration, select the first matching file.
    sens_map_path = random.choice(sens_map_files)
    
    # Load sensitivity map
    sens_maps = load_sensitivity_map(sens_map_path)
    
    # Visualize the sensitivity maps and their FFTs
    visualize_sensitivity_map(sens_maps, sens_map_path.stem, folder)

if __name__ == "__main__":
    main()
