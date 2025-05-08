import torch.nn as nn
import torch

def attention(query, key, value, mask = None):
    """Compute 'Scaled Dot Product Attention'.

    Args:
        query (Tensor): Query tensor
        key (Tensor): Key tensor
        value (Tensor): Value tensor
        mask (Tensor): Mask tensor
    """
    d_k = query.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
    if mask is not None:
        score += mask
    return torch.matmul(torch.softmax(score, dim=-1), value)

# -- test code
# query = torch.randn(2, 4, 3)
# key = torch.randn(2, 4, 3)
# value = torch.randn(2, 4, 3)
# print(attention(query, key, value))