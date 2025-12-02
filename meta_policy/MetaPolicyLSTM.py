import torch
import torch.nn as nn


class MetaPolicyLSTM(nn.Module):
    """
    HiLight Section 4.1.2 — LSTM-based Sub-goal Generation

    INPUT:
        subregion_seq : (B, T, M*d_reg)
            B = batch size
            T = time steps (20)
            M = number of regions (4)
            d_reg = regional embedding dim (4)
            So normally: (B, 20, 16)

    OUTPUT:
        G : (B, 1, d_reg)   # (B, 1, 4)
            Global sub-goal vector
    """

    def __init__(self, M=4, d_reg=4, hidden_size=256):
        super().__init__()

        self.M = M
        self.d_reg = d_reg
        self.hidden_size = hidden_size

        # LSTM input size = flattened regional features
        self.input_dim = M * d_reg   # 16

        # 4-layer LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=4,          # paper says 4-layer LSTM
            batch_first=True
        )

        # FC layer to project hidden states to 4-dim subgoal
        self.fc = nn.Linear(hidden_size, d_reg)

    def forward(self, subregion_seq):
        """
        subregion_seq: (B, T, M*d_reg)
        """

        B, T, F = subregion_seq.shape
        assert F == self.M * self.d_reg, "Incorrect input dimension"

        # ------------------------------------------------------------
        # 1. Feed into LSTM  → output: (B, T, hidden_size)
        # ------------------------------------------------------------
        lstm_out, (h_n, c_n) = self.lstm(subregion_seq)

        # ------------------------------------------------------------
        # 2. Extract last-step hidden state for each LSTM layer
        #    h_n shape = (num_layers, B, hidden_size)
        #    We want last layer → h_n[-1] = (B, hidden_size)
        # ------------------------------------------------------------
        final_hidden = h_n[-1]   # (B, hidden_size)

        # ------------------------------------------------------------
        # 3. FC projection to 4-dim global subgoal
        # ------------------------------------------------------------
        G = self.fc(final_hidden)   # (B, 4)

        # ------------------------------------------------------------
        # 4. Reshape to (B, 1, 4) to match table output (1, 1, 4)
        # ------------------------------------------------------------
        return G.unsqueeze(1)       # (B, 1, 4)
if __name__ == "__main__":
    print("\n=== Running MetaPolicyLSTM Test ===\n")

    lstm_model = MetaPolicyLSTM(M=4, d_reg=4, hidden_size=256)

    # Fake input from the Transformer
    # (batch=1, time=20, 4 regions * 4 dims = 16)
    subregion_seq = torch.randn(1, 20, 16)

    G = lstm_model(subregion_seq)

    print("Input shape:", subregion_seq.shape)
    print("Output G shape:", G.shape)     # (1, 1, 4)

    assert G.shape == (1, 1, 4)
