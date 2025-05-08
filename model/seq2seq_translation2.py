from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
from model.transformer.transformer import Transformer
from .token_embedding import TokenEmbedding
from .position_encoding import PositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.src_positional_encoding = PositionalEncoding(emb_size, maxlen=src_vocab_size, dropout=dropout)
        self.tgt_positionnal_encoding = PositionalEncoding(emb_size, maxlen=tgt_vocab_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_token_emb = self.src_tok_emb(src)
        src_pe_emb = self.src_positional_encoding(src)
        src_emb = src_token_emb + src_pe_emb

        tgt_token_emb = self.tgt_tok_emb(trg)
        tgt_pe_emb = self.tgt_positionnal_encoding(trg)
        tgt_emb = tgt_token_emb + tgt_pe_emb
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        # torch.onnx.export(self.transformer, (src_emb, tgt_emb, src_mask, tgt_mask), f"base_transformer.onnx")
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.src_positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.src_positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)