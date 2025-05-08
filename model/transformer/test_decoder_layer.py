import torch.nn as nn
from test_utils import clones
from test_multi_head_attention import MultiHeadsAttention
from test_feed_forward import FeedForward
from test_sublayer_connection import SublayerConnection

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, droput = 0.0):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadsAttention(heads, d_model, droput)
        self.src_attn = MultiHeadsAttention(heads, d_model, droput)
        self.feed_forward = FeedForward(d_model, d_ff, droput)
        self.sublayer = clones(SublayerConnection(d_model, droput), 3)

    def forward(self, x, memory, src_mask, tgt_mask, memory_mask = None, src_padding_mask = None, tgt_padding_mask = None, memory_key_padding_mask = None):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, tgt_padding_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, src_padding_mask))
        return self.sublayer[2](x, self.feed_forward)

