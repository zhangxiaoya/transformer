import torch
import math
import numpy as np
import torch.nn as nn
from torch import Tensor

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * torch.log(torch.tensor(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.encoding1 = nn.Embedding(maxlen,
                                     emb_size,
                                     _weight=pos_embedding,
                                     _freeze=True)
        self.encoding = nn.Embedding.from_pretrained(pos_embedding, freeze=True)
        # self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # import pdb; pdb.set_trace()
        return self.encoding(token_embedding.long())
        # return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])

# import matplotlib.pyplot as plt
# pe = PositionalEncoding(512, 0)
# y = pe(torch.zeros(128, 100, 512))
# plt.figure(figsize=(15, 5))
# plt.plot(y[0, :, 4:8].data.numpy())
# plt.legend(['dim %d' % p for p in [4, 5, 6, 7]])
# plt.savefig('position_encodings.png')
