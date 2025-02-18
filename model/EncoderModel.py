import torch
from torch import nn
import LSTMLayer
import EmbeddingLayer


class Encoder(nn.Module):
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
        super(Encoder, self).__init__()
        self.seq_encoder = LSTMLayer.SeqEncoder(
            embedding_size, hidden_size, num_layers, drop_out
        )
        self.emb = EmbeddingLayer.FullEmbedding(
            vocab_size, embedding_size, max_len, device, drop_out
        )
        self.decoder = nn.Linear(hidden_size * 2, num_class)

    def attention_layer(self, lstm_output: torch.Tensor, final_state: torch.Tensor):
        """attention层

        Args:
            lstm_output (torch.Tensor): lstm的输出，经过变换到attention层，为[batch,seq_len,hidden_size*D]
            final_state (torch.Tensor): lstm最后一个时间步的输出。[num_layers*D,batch,hidden_size]
        """
        num_layers = final_state.size(0)/2
        hidden_size = final_state.size(2)
        hidden = final_state.view(-1,hidden_size*2,num_layers)
        attn_weights = torch.bmm(lstm_output,hidden) # [batch_size, seq_len,num_layers]
        pass
