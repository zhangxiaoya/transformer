import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(tgt, memory, tgt_mask, src_mask)

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask = None, src_padding_mask = None, tgt_padding_mask = None, memory_key_padding_mask = None):
        memory = self.encoder(src, src_padding_mask)
        return self.decoder(tgt, memory, tgt_mask, src_mask, memory_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)