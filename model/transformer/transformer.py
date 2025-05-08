from model.transformer.encoder_decoder import EncoderDecoder
from model.transformer.encoder import Encoder
from model.transformer.decoder import Decoder
from model.transformer.generator import Generator
from model.transformer.encoder_layer import EncoderLayer
from model.transformer.decoder_layer import DecoderLayer
from model.transformer.multi_head_attention import MultiHeadAttention
from model.transformer.feed_forward import FeedForward
import copy
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(Transformer, self).__init__()
        # self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        # self._src_attn = MultiHeadAttention(d_model, nhead, dropout)
        # self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        # self.encoder_layer = EncoderLayer(d_model, self_attn, self.feed_forward, dropout)
        # self.encoder = Encoder(self.encoder_layer, num_encoder_layers)
        # self.decoder_layer = DecoderLayer(d_model, self_attn, self._src_attn, self.feed_forward, dropout)
        # self.decoder = Decoder(self.decoder_layer, num_decoder_layers)
        # self.encoder_decoder = EncoderDecoder(self.encoder, self.decoder)
        self.make_model(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)


    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask = None, src_padding_mask = None, tgt_padding_mask = None, memory_key_padding_mask = None):
        return self.encoder_decoder(src, tgt, src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)


    def make_model(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        """
        Construct a model object based on the Transformer architecture.
        """
        c = copy.deepcopy
        self.self_attn = MultiHeadAttention(nhead, d_model, dropout)
        self._src_attn = MultiHeadAttention(nhead, d_model, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

        self.encoder_layer = EncoderLayer(d_model, c(self.self_attn), c(self.feed_forward), dropout)
        self.encoder = Encoder(self.encoder_layer, num_encoder_layers)
        self.decoder_layer = DecoderLayer(d_model, c(self.self_attn), c(self._src_attn), c(self.feed_forward), dropout)
        self.decoder = Decoder(self.decoder_layer, num_decoder_layers)
        self.encoder_decoder = EncoderDecoder(self.encoder, self.decoder)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.encoder_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)