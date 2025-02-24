import torch
from torch import nn
from torch.nn import functional as F
from . import LSTMLayer,EmbeddingLayer

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
        self.key_matrix = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.query = nn.Parameter(torch.Tensor(hidden_size * 2))
        # 初始化矩阵参数
        nn.init.uniform_(self.key_matrix, -0.1, 0.1)
        nn.init.uniform_(self.query, -0.1, 0.1)
        # 解码器，输出class
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size, num_class),
        )

    def forward(self, x):
        emb = self.embedding(x)  # [batch_size,seq_len,emb_size]
        emb = emb.permute(1, 0, 2)  # [seq_len,batch_size,emb_size]
        h0 = torch.zeros(self.num_layers * 2, len(x), self.hidden_size).to(device=self.device)
        c0 = torch.zeros(self.num_layers * 2, len(x), self.hidden_size).to(device=self.device)
        outputs, (h, c) = self.seq_encoder(
            emb, h0, c0
        )  # outputs: [seq_len,batch_size,hidden_size*2]
        outputs = outputs.permute(1, 0, 2)  # [batch_size,seq_len,hidden_size*2]
        key = torch.tanh(
            torch.matmul(outputs, self.key_matrix)
        )  # [seq_len,batch_size,hidden_size*2]

        # torch.matmul(key,self.query)的结果为 [batch_size,seq_len]因为做的是内积
        # 再对第1维做softmax
        score = F.softmax(torch.matmul(key, self.query), dim=1).unsqueeze(
            -1
        )  # [batch_size,seq_len,1]

        out = outputs * score  # [batch_size,seq_len,hidden_size*2]
        out = torch.sum(out, dim=1)  # [batch_size,hidden_size*2]
        out = F.gelu(out)
        final_outputs = self.decoder(out)
        return final_outputs
