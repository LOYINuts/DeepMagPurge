import torch.nn as nn
import torch


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

    def forward(self, x, hidden_state, cell_state):
        output, (final_hidden_state, final_cell_state) = self.lstm(
            x, (hidden_state, cell_state)
        )
        return output, (final_hidden_state, final_cell_state)
