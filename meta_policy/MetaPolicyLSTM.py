import torch
import torch.nn as nn

class MetaPolicyLSTM(nn.Module):
    """
    Modified for your setup:
    Output shape = (B, 4, 4)
    One 4-dim subgoal per region.
    """

    def __init__(self, M=4, d_reg=4, hidden_size=256):
        super().__init__()

        self.M = M
        self.d_reg = d_reg
        self.hidden_size = hidden_size

        # LSTM input: flattened regional embeddings (16)
        self.input_dim = M * d_reg

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            num_layers=4,
            batch_first=True
        )

        # FC layer to produce per-region subgoal vectors
        self.fc = nn.Linear(hidden_size, M * d_reg)

    def forward(self, subregion_seq):
        """
        subregion_seq: (B, T, 16)
        Output: (B, 4, 4)
        """

        B, T, F = subregion_seq.shape
        assert F == self.M * self.d_reg

        lstm_out, (h_n, _) = self.lstm(subregion_seq)

        # Last layer hidden state → (B, 256)
        h_last = h_n[-1]

        # FC → (B, 16)
        out = self.fc(h_last)

        # Reshape → (B, 4, 4)
        return out.view(B, self.M, self.d_reg)


if __name__ == "__main__":
    lstm = MetaPolicyLSTM()

    fake = torch.randn(1, 20, 16)
    G = lstm(fake)

    print("Output shape:", G.shape)   # (1, 4, 4)