import torch.nn as nn
import torch


class KmerEmbedding(nn.Module):
    """Kmer,单词embedding"""

    def __init__(self, vocab_size: int, d_model: int):
        """初始化

        Args:
            vocab_size (int): 词表大小
            d_model (int): 模型维度
        """
        super(KmerEmbedding, self).__init__()
        self.encoder = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.encoder(x)


class PositionEmbedding(nn.Module):
    """位置编码"""

    def __init__(self, max_len: int, d_model: int, device: str):
        """初始化

        Args:
            max_len(int): 标量，序列最大长度
            d_model(int): 标量，模型维度
            device(str): str，训练设备
        """
        super(PositionEmbedding, self).__init__()
        # 初始化0矩阵
        self.encoding = torch.zeros(max_len, d_model, device=device)
        # 位置编码不需要优化，就不需要梯度更新
        self.encoding.requires_grad = False
        # 定义pos，生成位置索引
        pos = torch.arange(0, max_len)
        pos = pos.to(device)
        # 类型转换为浮点型便于计算，在进行维度拓展为二维张量,利用广播机制自动对其
        pos = pos.float().unsqueeze(dim=1)
        # 根据公式计算
        frequencies_indices = torch.arange(0, d_model, step=2, device=device).float()
        frequencies = 1.0 / torch.pow(
            10000.0, frequencies_indices // d_model
        ).unsqueeze(dim=0)
        self.encoding[:, 0::2] = torch.sin(pos * frequencies)
        self.encoding[:, 1::2] = torch.cos(pos * frequencies)

    def forward(self, x):
        # 获取批量大小和序列长度
        seq_len = x.size(1)
        return self.encoding[:, :seq_len]


class FullEmbedding(nn.Module):
    """embedding层"""

    def __init__(
        self, vocab: int, d_model: int, max_len: int, drop_prob: float, device: str
    ):
        super(FullEmbedding, self).__init__()
        self.token_emb = KmerEmbedding(vocab, d_model)
        self.pos_emb = PositionEmbedding(max_len, d_model, device)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop(tok_emb + pos_emb)
