import torch.nn as nn
from model.transformer.utils import clones
from model.transformer.sublayer_connection import SublayerConnection

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sub_layers = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, memory_mask = None, src_padding_mask = None, tgt_padding_mask = None, memory_key_padding_mask = None):
        m = memory
        x = self.sub_layers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, tgt_padding_mask))
        x = self.sub_layers[1](x, lambda x: self.src_attn(x, m, m, padding_mask=src_padding_mask))
        return self.sub_layers[2](x, self.feed_forward)