import torch
from torch import nn
from torch.nn import functional as F
from . import LSTMLayer, EmbeddingLayer,AttentionLayer


class TaxonModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        device: str,
        max_len: int,
        num_layers: int,
        num_class: int,
        d_model: int,
        row: int,
        drop_out: float = 0.5,
    ):
        super(TaxonModel, self).__init__()
        self.num_layers = num_layers
        self.num_class = num_class
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.device = device
        self.max_len = max_len
        self.drop_out = drop_out
        self.seq_encoder = LSTMLayer.SeqEncoder(
            embedding_size, hidden_size, num_layers, drop_out
        )
        self.embedding = EmbeddingLayer.FullEmbedding(
            vocab_size, embedding_size, max_len, device, drop_out
        )
        # attention相关
        self.attention = AttentionLayer.Attention(hidden_size*2)
        # 解码器，输出class
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size, num_class),
        )

    def forward(self, x):
        x = self.embedding(x)  # [batch_size,seq_len,emb_size]
        x = x.permute(1, 0, 2)  # [seq_len,batch_size,emb_size]
        x = self.seq_encoder(x)  # x: [seq_len,batch_size,hidden_size*2]
        x = x.permute(1, 0, 2)  # [batch_size,seq_len,hidden_size*2]
        x = self.attention(x)
        final_outputs = self.decoder(x)
        return final_outputs
