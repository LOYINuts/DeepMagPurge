from torch import nn
from . import LSTMLayer,EmbeddingLayer,AttentionLayer


class TaxonModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
        device,
        max_len: int,
        num_layers: int,
        num_class: int,
        drop_out: float = 0.5,
    ):
        """TaxonModel 物种分类模型，结合了嵌入层、LSTM 编码器、注意力机制和解码器。

        Args:
            vocab_size (int): 词汇表的大小。
            embedding_size (int): 嵌入向量的维度。
            hidden_size (int): LSTM 隐藏层的维度。
            device: 模型运行的设备。
            max_len (int): 输入序列的最大长度。
            num_layers (int): LSTM 层的数量。
            num_class (int): 分类的类别数量。
            drop_out (float, 可选): 丢弃率，默认为 0.5。
        """
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
        self.attention = AttentionLayer.Attention(hidden_size * 2)
        # 解码器，输出class
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size, num_class),
        )

    def forward(self, x):
        """
        前向传播方法，定义了输入数据在模型中的流动过程。

        Args:
            x (torch.Tensor): 输入的张量，形状为 [batch_size, seq_len]。

        Returns:
            torch.Tensor: 模型的输出，形状为 [batch_size, num_class]。
        """
        x = self.embedding(x)  # [batch_size,seq_len,emb_size]
        x = x.permute(1, 0, 2)  # [seq_len,batch_size,emb_size]
        x = self.seq_encoder(x)  # x: [seq_len,batch_size,hidden_size*2]
        x = x.permute(1, 0, 2)  # [batch_size,seq_len,hidden_size*2]
        x = self.attention(x)
        final_outputs = self.decoder(x)
        return final_outputs
