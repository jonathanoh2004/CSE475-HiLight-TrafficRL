import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  Positional Encoding (Sinusoidal) — Matches HiLight Equation
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with configurable scaling parameter.
    Matches HiLight paper: 
        PE(pos, 2i)   = sin(pos / τ^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / τ^(2i/d_model))

    Input / Output shape:
        (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, max_len: int, tau: float = 10000.0):
        super().__init__()

        pe = torch.zeros(max_len, d_model)      # (seq_len, d)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(tau) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)   # even dims
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dims

        self.register_buffer("pe", pe.unsqueeze(0))    # (1, seq_len, d)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ============================================================
#  Transformer Layer (No Layer Normalization) — Matches HiLight
# ============================================================

class TransformerEncoderLayerNoNorm(nn.Module):
    """
    A Transformer encoder layer WITHOUT LayerNorm.

    HiLight paper explicitly states "no layer norm" in Table 6.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout_ff = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, src):
        # --- Multi-head Self Attention ---
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout_attn(attn_output)

        # --- Feedforward ---
        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(src))))
        src = src + ff

        return src


# ============================================================
#  Meta-Policy Transformer (HiLight §4.1.1)
# ============================================================

class MetaPolicyTransformer(nn.Module):
    """
    HiLight Transformer Encoder over Subregions.

    INPUT:
        regional_states: (B, T, M, d_reg)
            B = batch size
            T = time steps (20)
            M = number of subregions (4)
            d_reg = feature per region (4)

    OUTPUT:
        F_g : (B, T, d_reg)
            Global feature per timestep (comes from the class token)

        subregion_seq : (B, T, M * d_reg)
            Flattened regional features (used for LSTM)
    """

    def __init__(
        self,
        d_reg=4,
        n_regions=4,
        n_layers=3,
        n_heads=2,
        d_ff=16,
        tau=10000.0,
        dropout=0.1,
    ):
        super().__init__()

        self.d_reg = d_reg
        self.n_regions = n_regions
        self.seq_len = n_regions + 1  # 1 global token + 4 subregions = 5 tokens

        # ===== Learnable class token (x_token in HiLight) =====
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_reg))

        # ===== Positional encoding (with τ scaling) =====
        self.pos_encoding = PositionalEncoding(d_reg, self.seq_len, tau)

        # ===== Transformer layers (no layer norm!) =====
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayerNoNorm(
                    d_model=d_reg,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    # ----------------------------------------------------------
    # Forward Pass
    # ----------------------------------------------------------

    def forward(self, regional_states):
        """
        regional_states: (B, T, M, d_reg)
        """

        B, T, M, D = regional_states.shape
        assert M == self.n_regions
        assert D == self.d_reg

        # Merge batch and time: treat each timestep independently
        x = regional_states.view(B * T, M, D)  # (B*T, 4, 4)

        # Prepend learnable class token
        cls = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B*T, 5, 4)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Run through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Extract global token (index 0)
        global_tokens = x[:, 0, :]  # (B*T, 4)

        # Extract regional tokens (1..M)
        regional_tokens = x[:, 1:, :]  # (B*T, 4, 4)

        # Reshape back to (B, T, ...)
        F_g = global_tokens.view(B, T, D)  # (B, 20, 4)
        subregion_seq = regional_tokens.view(B, T, M * D)  # (B, 20, 16)

        return F_g, subregion_seq


# ============================================================
#  Test Block — RUN THIS FILE DIRECTLY TO TEST
# ============================================================

if __name__ == "__main__":
    print("\n=== Running MetaPolicyTransformer Test ===\n")

    model = MetaPolicyTransformer(
        d_reg=4,
        n_regions=4,
        n_layers=3,
        n_heads=2,
        d_ff=16,
    )

    # Paper example: (batch=2, T=20, M=4, d_reg=4)
    regional_states = torch.randn(1, 20, 4, 4)

    F_g, subregion_seq = model(regional_states)

    print("Input shape:             ", regional_states.shape)
    print("Global feature shape:    ", F_g.shape)        # Expected (2, 20, 4)
    print("Subregion sequence shape:", subregion_seq.shape)  # Expected (2, 20, 16)

    assert F_g.shape == (1, 20, 4)
    assert subregion_seq.shape == (1, 20, 16)

