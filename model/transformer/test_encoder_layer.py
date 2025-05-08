import torch.nn as nn
from test_utils import clones
from multi_head_attention import MultiHeadAttention
from test_feed_forward import FeedForward
from model.transformer.test_sublayer_connection import SublayerConnection


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(heads, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, padding_mask):
        x = self.sublayer[0](x,lambda x: self.self_attn(x, x, x, padding_mask))
        return self.sublayer[1](x, self.feed_forward)