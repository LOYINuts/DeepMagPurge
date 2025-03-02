import torch.nn as nn


class SeqEncoder(nn.Module):
    def __init__(
        self, embbeding_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.5
    ):
        super(SeqEncoder, self).__init__()
        self.embbeding_dim = embbeding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=embbeding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return output
