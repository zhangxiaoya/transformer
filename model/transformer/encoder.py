import torch.nn as nn
from model.transformer.utils import clones
# from model.transformer.layer_norm import LayerNorm
# import torch.nn.LayerNorm as LayerNorm

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, padding_mask = None):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return self.norm(x)