import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        attn_weights = self.attn(lstm_output).squeeze(2)  # [batch_size, seq_len]
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size,seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context
