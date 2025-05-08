import torch
import math

def attention(query, key, value, mask=None):
    d_k = key.size(-1)
    score = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)
    score = torch.nn.functional.softmax(score, dim = -1)
    att = torch.matmul(score, value)
    return att, score

# --------------------------------------------------
# Test Attention
# --------------------------------------------------
# batch_size = 64
# seq_len = 10
# d_k = 32
# query = torch.rand(batch_size, seq_len, d_k)
# key = torch.rand(batch_size, seq_len, d_k)
# value = torch.rand(batch_size, seq_len, d_k)
# mask = None
# att, score = attention(query, key, value, mask)
# --------------------------------------------------