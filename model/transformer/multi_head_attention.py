import torch.nn as nn
import torch
from model.transformer.attention import attention
from model.transformer.utils import clones

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, batch_first = False):
        """_summary_

        Args:
            h (_type_): _description_
            d_model (_type_): _description_
            dropout (float, optional): _description_. Defaults to 0.1.
            batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h ==0, f"d_model must be a multiple of h, detected d_model: {d_model}, h: {h}"
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # WQ, WK, WV, WO
        self.batch_first = batch_first

    def forward(self, query, keys, value, attn_mask=None, padding_mask = None):
        mask = None
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(-2).unsqueeze(-2).float()
            padding_mask = padding_mask.masked_fill(padding_mask == 1, float('-inf'))
            mask = padding_mask
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            if mask is None:
                mask = attn_mask
            else:
                mask = mask + attn_mask
        if self.batch_first is False:
            query, keys, value = [x.permute(1, 0, 2) for x in (query, keys, value)]
        b_batch = query.size(0)
        query, keys, value = [l(x) for l, x in zip(self.linears, (query, keys, value))]
        query, keys, value = [x.view(b_batch, -1, self.h, self.d_k).transpose(1, 2) for x in (query, keys, value)]
        x = attention(query, keys, value, mask)
        x = x.transpose(1, 2).contiguous().view(b_batch, -1, self.h * self.d_k)
        if self.batch_first is False:
            x = x.permute(1, 0, 2)
        return self.linears[-1](x)