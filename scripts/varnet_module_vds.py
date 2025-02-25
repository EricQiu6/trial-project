"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms

from unet import Unet


class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        latent_dim: int = 512,  # New: latent vector dimension
    ):
        super().__init__()
        self.latent_dim = latent_dim
        # Project latent vector to match U-Net input channels
        self.latent_proj = nn.Linear(latent_dim, in_chans)
        self.unet = Unet(
            in_chans=in_chans * 2,  # Double channels for concatenation
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )
        # self.latent_proj.weight = torch.nn.init.normal_(self.latent_proj.weight, mean=0.0, std=0.01)
        # self.latent_proj.bias = torch.nn.init.constant_(self.latent_proj.bias, 0.0)
        # print("NormUnet - Latent projection (nn.linear) initialized weights min/max:", self.latent_proj.weight.min().item(), self.latent_proj.weight.max().item())
        # print("NormUnet - Latent projection (nn.linear) initialized bias min/max:", self.latent_proj.bias.min().item(), self.latent_proj.bias.max().item())
        if torch.isnan(self.latent_proj.weight).any():
            print("ERROR: Latent projection weights contain NaN!")
            exit(1)
        if torch.isnan(self.latent_proj.bias).any():
            print("ERROR: Latent projection bias contains NaN!")
            exit(1)

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        
        # print("NormUnet - Latent projection (nn.linear) weights min/max:", self.latent_proj.weight.min().item(), self.latent_proj.weight.max().item())
        if torch.isnan(self.latent_proj.weight).any():
            print("ERROR: Latent projection weights contain NaN!")
            exit(1)

        
        # print("NormUnet - Input min/max:", x.min().item(), x.max().item())

        # Convert complex to channel dimension [batch, 2*coils, height, width]
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        # print("NormUnet - After norm min/max:", x.min().item(), x.max().item())

        x, pad_sizes = self.pad(x)
        # print("NormUnet - After padding min/max:", x.min().item(), x.max().item())
        if torch.isnan(x).any():
            print("ERROR: Input tensor 'x' contains NaN!")
            exit(1)

        # print("Latent vector min/max:", latent_vector.min().item(), latent_vector.max().item())
        if torch.isnan(latent_vector).any():
            print("ERROR: Latent vector contains NaN!")
            exit(1)

        # Process latent vector: project and reshape to match spatial dimensions
        latent_features = self.latent_proj(latent_vector)  # [batch, in_chans]
        # print("NormUnet - After latent projection min/max:", latent_features.min().item(), latent_features.max().item())
        if torch.isnan(latent_features).any():
            print("ERROR: Latent features contain NaN!")
            exit(1)

        latent_features = latent_features.view(*latent_features.shape, 1, 1)  # [batch, in_chans, 1, 1]
        latent_features = latent_features.expand(-1, -1, x.shape[-2], x.shape[-1])  # [batch, in_chans, H, W]
        # print("Latent features(expanded) min/max:", latent_features.min().item(), latent_features.max().item())
        if torch.isnan(latent_features).any():
            print("ERROR: Latent features contain NaN!")
            exit(1)
        

        # print("Latent features shape:", latent_features.shape)
        # print("Input tensor shape before concat:", x.shape)



        # Concatenate latent features with input
        x = torch.cat([x, latent_features], dim=1)  # [batch, in_chans*2, H, W]
        # print("NormUnet - After latent concat min/max:", x.min().item(), x.max().item())

        x = self.unet(x)
        # print("NormUnet - U-Net output min/max:", x.min().item(), x.max().item())

        x = self.unpad(x, *pad_sizes)
        # print("NormUnet - After unpadding min/max:", x.min().item(), x.max().item())

        x = self.unnorm(x, mean, std)
        # print("NormUnet - After unnorm min/max:", x.min().item(), x.max().item())

        x = self.chan_complex_to_last_dim(x)
        # print("NormUnet - After chan_complex_to_last_dim min/max:", x.min().item(), x.max().item())

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        # convert to image space
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )


class LatentVarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        # self.sens_net = SensitivityModel(
        #     chans=sens_chans,
        #     num_pools=sens_pools,
        #     mask_center=mask_center,
        # )
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(self, masked_kspace, mask, latent_vector, sens_maps):
        # print("Forward pass started...")
        
        # print("Step 1 - Input k-space min/max:", masked_kspace.min().item(), masked_kspace.max().item())
        # print("Step 2 - Latent vector min/max:", latent_vector.min().item(), latent_vector.max().item())

        kspace_pred = masked_kspace.clone()

        for idx, cascade in enumerate(self.cascades):
            prev_kspace = kspace_pred.clone() # Save previous k-space for comparison

            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps, latent_vector)

            # print(f"Cascade {idx} - Before/After min/max:")
            # print("Before:", prev_kspace.min().item(), prev_kspace.max().item())
            # print("After:", kspace_pred.min().item(), kspace_pred.max().item())
            # print("Difference min/max:", (kspace_pred - prev_kspace).min().item(), (kspace_pred - prev_kspace).max().item())

            if torch.allclose(prev_kspace, kspace_pred, atol=1e-8, equal_nan=True):  # Check if k-space is actually changing
                print(f"WARNING: Cascade {idx} did not modify k-space!")
            
            # from here we know that cascade changed kspace_pred into all nans
            # print("kspace_pred itself: ", kspace_pred)
            # print("prev_kspace itself: ", prev_kspace)

            if torch.isnan(kspace_pred).any():
                print(f"NaN detected in cascade {idx}")
                exit(1)

        recon = fastmri.ifft2c(kspace_pred)
        # print("Step 3 - IFFT recon min/max:", recon.min().item(), recon.max().item())

        abs_recon = fastmri.complex_abs(recon)
        # print("Step 4 - Absolute recon min/max:", abs_recon.min().item(), abs_recon.max().item())

        final_output = fastmri.rss(abs_recon, dim=1)
        # print("Step 5 - Final output min/max:", final_output.min().item(), final_output.max().item())

        return final_output


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        # print("Initial DC Weight:", self.dc_weight.item())


    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        # print("inside sens_reduce - current_kspace min/max:", x.min().item(), x.max().item())
        if torch.isnan(x).any():
            print("ERROR: current_kspace contains NaN before `ifft2c`!")
            exit(1)

        ifft_output = fastmri.ifft2c(x)
        # print("sens_reduce - ifft2c output min/max:", ifft_output.min().item(), ifft_output.max().item())
        if torch.isnan(ifft_output).any():
            print("ERROR: `ifft2c(x)` produced NaNs!")
            exit(1)

        # print("sens_reduce - sens_maps min/max:", sens_maps.min().item(), sens_maps.max().item())
        if torch.isnan(sens_maps).any():
            print("ERROR: `sens_maps` contains NaNs!")
            exit(1)

        print("ifft_output shape:", ifft_output.shape)
        print("sens_maps shape:", sens_maps.shape)
        if ifft_output.shape != sens_maps.shape:
            print("ERROR: ifft_output and sens_maps shapes do not match!")
            print("ifft_output shape:", ifft_output.shape)
            print("sens_maps shape:", sens_maps.shape)
            exit(1)

        complex_mul_output = fastmri.complex_mul(ifft_output, fastmri.complex_conj(sens_maps))
        # print("sens_reduce - complex_mul output min/max:", complex_mul_output.min().item(), complex_mul_output.max().item())
        if torch.isnan(complex_mul_output).any():
            print("ERROR: `complex_mul(ifft_output, fastmri.complex_conj(sens_maps))` produced NaNs!")
            exit(1)

        return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)

    def forward(self, current_kspace, ref_kspace, mask, sens_maps, latent_vector):
        # print("VarNet Block - Before DC, current_kspace min/max:", current_kspace.min().item(), current_kspace.max().item())
        # print("VarNet Block - Before DC, ref_kspace min/max:", ref_kspace.min().item(), ref_kspace.max().item())
        # # this difference make sense since we start with clones
        # print("VarNet Block - Before DC, k-space difference min/max:", (current_kspace - ref_kspace).min().item(), (current_kspace - ref_kspace).max().item())
    

        # print("DC mask sum:", mask.sum().item())
        # print("DC weight min/max:", self.dc_weight.min().item(), self.dc_weight.max().item())


        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight

        # print("VarNet Block - After DC, soft_dc min/max:", soft_dc.min().item(), soft_dc.max().item())

        model_input = self.sens_reduce(current_kspace, sens_maps)
        # print("VarNet Block - Model input/sens_reduce min/max:", model_input.min().item(), model_input.max().item())

        model_output = self.model(model_input, latent_vector)
        # print("VarNet Block - Model output min/max:", model_output.min().item(), model_output.max().item())

        sens_expanded = self.sens_expand(model_output, sens_maps)
        # print("VarNet Block - Sens expanded min/max:", sens_expanded.min().item(), sens_expanded.max().item())

        output = current_kspace - soft_dc - sens_expanded
        # print("VarNet Block - Final output min/max:", output.min().item(), output.max().item())

        return output
