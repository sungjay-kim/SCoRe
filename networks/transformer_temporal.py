import torch
import torch.nn as nn
from typing import Optional, Tuple

import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Encodes continuous time 't' into a vector of size 'dim'.
    Uses sine/cosine frequencies: [sin(w0*t), cos(w0*t), sin(w1*t), cos(w1*t)...]
    where w_k = 1 / (10000 ** (2k / dim))
    """
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # Pre-compute frequencies if we wanted fixed ones, but simple dynamic computation 
        # is fast enough and flexible for any sequence length.
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
           t: (Batch, SeqLen) - Float timestamps (e.g. days)
        Returns:
           emb: (Batch, SeqLen, Dim)
        """
        B, L = t.shape
        half_dim = self.dim // 2
        
        # Generate frequencies
        exponent = torch.arange(half_dim, device=t.device, dtype=torch.float32) * -math.log(self.max_period) / half_dim
        freqs = torch.exp(exponent) # (HalfDim,)
        
        # Outer product: t.unsqueeze(-1) * freqs.unsqueeze(0)
        # (B, L, 1) * (1, 1, HalfDim) -> (B, L, HalfDim)
        args = t.unsqueeze(-1) * freqs.view(1, 1, -1)
        
        # Sin/Cos
        emb_sin = torch.sin(args)
        emb_cos = torch.cos(args)
        
        # Concat -> (B, L, Dim)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)
        
        # Handle odd dimension edge case (pad with 0)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(B, L, 1, device=t.device)], dim=-1)
            
        return emb

class CausalTemporalTransformer(nn.Module):
    """
    A Transformer-based Temporal Fusion Module that attends to the entire history
    but respects causality (can't see future frames).
    
    Structure:
    - Input Embedding: Project z_t (32) -> hidden_dim (e.g. 128)
    - Time Embedding: Add sinusoidal or learned embedding for delta_t
    - Transformer Encoder: N layers of Self-Attention with Causal Mask
    - Output Head: Project hidden_dim -> z_t_next (32)
    """
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        zero_init: bool = True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.zero_init = zero_init
        
        # 1. Input Projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # 2. Positional Encoding (Learned Index) + Time Encoding (Sinusoidal)
        # We use a simple learned embedding for position in sequence to denote ORDER
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        # We use Sinusoidal embedding to denote DURATION
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-Norm usually stabilizes deep transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Projection
        # Predicts the REFINED latent for the current timestep
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Init
        self._init_weights()
        
    def _init_weights(self):
        # Zero-init output projection to act as identity residual initially
        # This ensures the model starts with z_final = z_in + 0
        if self.zero_init:
            # Use small random initialization instead of strict zero to avoid dead gradients
            nn.init.uniform_(self.output_proj.weight, -0.01, 0.01)
            nn.init.uniform_(self.output_proj.bias, -0.01, 0.01)

    def forward(self, sequence_latents: torch.Tensor, sequence_times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            sequence_latents: (Batch, SeqLen, LatentDim) - All frames seen so far
            sequence_times: (Batch, SeqLen) - Relative timestamps or None
            
        Returns:
            processed_sequence: (Batch, SeqLen, LatentDim)
        """
        B, L, D = sequence_latents.shape
        
        # 1. Embed Inputs
        x = self.input_proj(sequence_latents) # (B, L, Hidden)
        
        # 2. Add Positional Embeddings
        # Example: Simple index-based positions [0, 1, 2...]
        # 2. Add Positional Embeddings
        # Example: Simple index-based positions [0, 1, 2...]
        positions = torch.arange(L, device=sequence_latents.device).unsqueeze(0).expand(B, L)
        positions = positions.clamp(max=self.pos_emb.num_embeddings - 1)
        pos_emb = self.pos_emb(positions)
        x = x + pos_emb
        
        # 2b. Add Time Embeddings (If provided)
        if sequence_times is not None:
             # sequence_times: (B, L)
             t_emb = self.time_emb(sequence_times)
             x = x + t_emb
        
        # TODO: If we want to use REAL delta_times, we would project sequence_times -> embedding
        # But for now, simple ordering is enough for causality.
        
        # 3. Create Causal Mask
        # Mask[i, j] = -inf if j > i (cannot attend to future)
        # nn.Transformer expects float mask: 0 (allow) or -inf (forbid)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(sequence_latents.device)
        
        # 4. Pass through Transformer
        # x is (B, L, Hidden) because batch_first=True
        x_out = self.transformer(x, mask=causal_mask)
        
        # 5. Project Back
        z_out = self.output_proj(x_out) # (B, L, Latent)
        
        # 6. Global Residual Connection (Identity Bias)
        # We want the transformer to learn the *residual* correction to the input measurements
        z_final = sequence_latents + z_out
        
        return z_final
