import torch
import torch.nn as nn
import math


class MetaPolicyTransformer(nn.Module):
    """
    HiLight Section 4.1.1 — Transformer Encoding Across Subregions

    Input shape: (B, T, M, d_reg)
    Output:
        F_g : (B, T, d_reg)   # global embedding from class token
        subregion_seq : (B, T, M*d_reg)  # flattened subregion embeddings
    """

    def __init__(self, d_reg=4, n_regions=4, n_layers=3, n_heads=2, ff_dim=165, tau=10000):
        super().__init__()

        self.d_reg = d_reg
        self.M = n_regions
        self.model_dim = d_reg
        self.tau = tau

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_reg))

        # Positional encoding for region axis
        self.register_buffer("pe", self._build_positional_encoding(self.M + 1, d_reg))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_reg,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    # -------------------------------------------------------------
    # Positional Encoding (HiLight formula)
    # -------------------------------------------------------------
    def _build_positional_encoding(self, length, dim):
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)

        div_term = self.tau ** (torch.arange(0, dim, 2).float() / dim)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        return pe  # shape (length, dim)

    # -------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------
    def forward(self, region_seq):
        """
        region_seq shape:
            (B, T, M, d_reg)

        Output:
            F_g : (B, T, d_reg)
            subregion_seq : (B, T, M*d_reg)
        """

        B, T, M, d_reg = region_seq.shape
        assert M == self.M

        # -----------------------------------------
        # 1. Prepend class token per timestep
        # -----------------------------------------
        cls = self.cls_token.expand(B, T, 1, d_reg)
        x = torch.cat([cls, region_seq], dim=2)   # (B, T, M+1, d_reg)

        # -----------------------------------------
        # 2. Add positional encoding over region axis
        # -----------------------------------------
        pe = self.pe.unsqueeze(0).unsqueeze(0)     # (1,1,M+1,d_reg)
        x = x + pe[:, :, : M+1]

        # -----------------------------------------
        # 3. Merge region axis into sequence axis for transformer
        # -----------------------------------------
        x = x.reshape(B * T, M + 1, d_reg)

        # -----------------------------------------
        # 4. Transformer Encoder
        # -----------------------------------------
        x = self.encoder(x)   # (B*T, M+1, d_reg)

        # -----------------------------------------
        # 5. Extract global embedding (CLS token)
        # -----------------------------------------
        F_g = x[:, 0, :]                    # (B*T, d_reg)
        F_g = F_g.reshape(B, T, d_reg)      # (B, T, d_reg)

        # -----------------------------------------
        # 6. Extract M subregion embeddings
        # -----------------------------------------
        sub = x[:, 1:, :].reshape(B, T, M * d_reg)

        return F_g, sub
