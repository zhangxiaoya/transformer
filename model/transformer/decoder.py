import torch.nn as nn
from model.transformer.utils import clones
# from model.transformer.layer_norm import LayerNorm
# import torch.nn.LayerNorm as LayerNorm

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, tgt_mask, src_mask=None, memory_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.norm(x)