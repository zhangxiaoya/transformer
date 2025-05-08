import torch.nn as nn

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        xx = self.norm(x)
        xx = sublayer(xx)
        xx = self.dropout(xx)
        return x + xx
