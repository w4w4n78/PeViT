import torch
import torch.nn as nn
from typing import Tuple


class PSPEncoder(nn.Module):
    """
    Periodic Shifted Patch (PSP) Encoder for raw EEG signals.
    
    Reference:
        Yunanto et al., "Periodic Shifted Patch on Vision Transformer for 
        Electroencephalogram Signal-Based Emotion Recognition", Algorithm 1.
    
    The PSP encoder enriches raw EEG signals on-the-fly while preserving spatial-temporal 
    relationships through overlapping shifted windowing.
    
    Args:
        num_channels (int): Number of EEG electrodes/channels (K).
        sampling_rate (int): EEG signal sampling rate (s).
        shifted_window (int): Shifted window parameter (w).
        patch_size (int): Temporal patch size (p_s).
    """
    def __init__(
        self,
        num_channels: int,
        sampling_rate: int,
        shifted_window: int,
        patch_size: int
    ):
        super(PSPEncoder, self).__init__()
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.shifted_window = shifted_window
        self.patch_size = patch_size
        self.patch_dim = num_channels * patch_size

    def get_num_patches(self, seq_len: int) -> Tuple[int, int, int]:
        """
        Calculates the total number of periodic shifts (c_x), partition window patches (c_p),
        and total patch tokens (n_p).
        """
        if seq_len < self.sampling_rate:
            raise ValueError(
                f"Sequence length ({seq_len}) must be greater than or equal to sampling rate ({self.sampling_rate})."
            )
        
        c_x = (seq_len - self.sampling_rate) // self.shifted_window + 1
        c_p = self.sampling_rate // self.patch_size
        n_p = c_x * c_p
        return c_x, c_p, n_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Periodic Shifted Patch construction.
        
        Args:
            x (torch.Tensor): Input raw EEG signals of shape (B, K, T)
                              B: Batch size
                              K: Channels (electrodes)
                              T: Temporal sequence length
                              
        Returns:
            torch.Tensor: Flattened patch tokens of shape (B, n_p, K * p_s)
        """
        B, K, T = x.shape
        c_x, c_p, n_p = self.get_num_patches(T)

        patches = []
        for ix in range(c_x):
            window_start = ix * self.shifted_window
            for jx in range(c_p):
                patch_start = window_start + jx * self.patch_size
                patch_end = patch_start + self.patch_size
                
                # Slice across all K channels: (B, K, p_s)
                patch = x[:, :, patch_start:patch_end]
                
                # Flatten spatial and temporal dimensions: (B, K * p_s)
                patch_flat = patch.contiguous().view(B, -1)
                patches.append(patch_flat)

        # Stack into sequence of patches: (B, n_p, K * p_s)
        patch_seq = torch.stack(patches, dim=1)
        return patch_seq
