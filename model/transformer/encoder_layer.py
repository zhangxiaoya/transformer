import torch.nn as nn
from model.transformer.utils import clones
from model.transformer.sublayer_connection import SublayerConnection

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, padding_mask):
        x = self.sublayer[0](x, lambda x : self.self_attn(x,x,x,padding_mask=padding_mask))
        return self.sublayer[1](x, self.feed_forward)