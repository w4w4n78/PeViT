import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .psp_encoder import PSPEncoder


class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer Encoder Layer with Layer Normalization, 
    Multi-Head Self-Attention (MSA), and Multi-Layer Perceptron (MLP).
    """
    def __init__(
        self,
        latent_dim: int,
        attn_heads: int,
        mlp_dim: int,
        dropout: float = 0.1
    ):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Attention with Residual Connection
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out

        # MLP with Residual Connection
        x = x + self.mlp(self.norm2(x))
        return x


class PeViT(nn.Module):
    """
    Periodic Shifted Patch Vision Transformer (PeViT) for EEG-based Emotion Recognition.
    
    Reference:
        Yunanto et al., "Periodic Shifted Patch on Vision Transformer for 
        Electroencephalogram Signal-Based Emotion Recognition"
        
    Args:
        num_channels (int): Number of EEG electrodes (K).
        sampling_rate (int): Sampling rate (s).
        shifted_window (int): PSP shifted window size (w).
        patch_size (int): PSP temporal patch size (p_s).
        sequence_length (int): Total EEG signal temporal length (T).
        latent_dim (int): Transformer latent embedding dimension (l).
        encoder_layers (int): Number of Transformer encoder layers (o).
        attn_heads (int): Number of attention heads in MSA.
        mlp_dim (int): Hidden dimension in feed-forward network (MLP).
        num_classes (int): Number of output emotion target classes.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        num_channels: int = 32,
        sampling_rate: int = 128,
        shifted_window: int = 16,
        patch_size: int = 32,
        sequence_length: int = 128,
        latent_dim: int = 128,
        encoder_layers: int = 3,
        attn_heads: int = 4,
        mlp_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super(PeViT, self).__init__()
        
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.shifted_window = shifted_window
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim

        # 1. Periodic Shifted Patch (PSP) Encoder Module
        self.psp_encoder = PSPEncoder(
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            shifted_window=shifted_window,
            patch_size=patch_size
        )

        _, _, self.num_patches = self.psp_encoder.get_num_patches(sequence_length)
        patch_dim = num_channels * patch_size

        # 2. EEG Token Projection & Embedding
        self.patch_projection = nn.Linear(patch_dim, latent_dim)
        
        # Learnable Class Token T_0
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        # Learnable Positional Embedding E_pos: (1, n_p + 1, latent_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, latent_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # 3. EEG-based Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                latent_dim=latent_dim,
                attn_heads=attn_heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(encoder_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)

        # 4. Classification Layer
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PeViT.
        
        Args:
            x (torch.Tensor): Raw EEG signal of shape (B, K, T)
            
        Returns:
            torch.Tensor: Emotion class logits of shape (B, num_classes)
        """
        B = x.shape[0]

        # Step 1: PSP Augmentation & Patch Extraction -> (B, n_p, K * p_s)
        patches = self.psp_encoder(x)

        # Step 2: Linear Projection -> (B, n_p, latent_dim)
        tokens = self.patch_projection(patches)

        # Prepend Class Token -> (B, n_p + 1, latent_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat((cls_tokens, tokens), dim=1)

        # Add Positional Embedding & Apply Dropout
        x_emb = x_emb + self.pos_embedding
        x_emb = self.dropout(x_emb)

        # Step 3: Transformer Encoder Layers
        for block in self.blocks:
            x_emb = block(x_emb)

        x_emb = self.norm(x_emb)

        # Step 4: Extract Class Token Representation -> (B, latent_dim)
        cls_out = x_emb[:, 0]

        # Classification Head -> (B, num_classes)
        logits = self.classifier(cls_out)
        return logits
