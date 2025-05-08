import torch
import torch.nn as nn
from test_attention import attention
from test_utils import clones

class MultiHeadsAttention(nn.Module):
    def __init__(self, h, d_model, dropout = 0.0, batch_first = False):
        super(MultiHeadsAttention, self).__init__()
        assert d_model % h == 0, f"d_model must be a multiple of h, detected d_model: {d_model}, h: {h}"
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, h * self.d_k), 4)
        self.batch_first = batch_first

    def forward(self, query, key, value, mask = None):
        query, keys, values = [l(x) for l, x in zip(self.linears, (query, key, value))]
        if self.batch_first is False:
            query, keys, values = [x.permute(1, 0, 2) for x in (query, keys, values)]
        b_batch = query.size(0)
        seq_len = query.size(1)
        query, key, value = [x.view(b_batch, seq_len, self.h, self.d_k).transpose(1,2) for x in (query, keys, values)]
        x, score, = attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(b_batch, seq_len, -1)
        if self.batch_first is False:
            x = x.permute(1, 0, 2)
        return self.linears[-1](x)
# --------------------------------------------------
# Test MultiHeadsAttention
# --------------------------------------------------
# batch_size = 64
# seq_len = 10
# d_k = 32
# h = 8
# query = torch.rand(batch_size, seq_len, d_k * h)
# key = torch.rand(batch_size, seq_len, d_k * h)
# value = torch.rand(batch_size, seq_len, d_k * h)
# mask = None
# att = MultiHeadsAttention(h, d_k * h)(query, key, value, mask)
# print(att.size())
# --------------------------------------------------