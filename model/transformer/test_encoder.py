import torch
import torch.nn as nn
from test_utils import clones
from test_encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x, padding_mask = None):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x
