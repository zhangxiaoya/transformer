import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 encoder_layer: nn.Modules,
                 num_layers: int):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask):
        for layer in self.layers:
            src = layer(src, mask)
        return src

class Decoder(nn.Module):
    def __init__(self,
                 decoder_layer: nn.Modules,
                 num_layers: int):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask)
        return tgt


class Transformer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 h_head: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super(Transformer, self).__init__()

    