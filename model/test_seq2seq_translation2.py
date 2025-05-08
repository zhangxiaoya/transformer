import torch
import torch.nn as nn
from torch.nn import Transformer
from .token_embedding import TokenEmbedding
from .position_encoding import PositionalEncoding

class Seq2SeqTranslation(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_model: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTranslation, self).__init__()

        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.src_token_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_token_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.src_position_encoding = PositionalEncoding(d_model, dropout, src_vocab_size)
        self.tgt_position_encoding = PositionalEncoding(d_model, dropout, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_tok_emb = self.src_token_embedding(src)
        src_pe_emb = self.src_position_encoding(src)
        src_emb = src_tok_emb + src_pe_emb
        tgt_tok_emb = self.tgt_token_embedding(tgt)
        tgt_pe_emb = self.tgt_position_encoding(tgt)
        tgt_emb = tgt_tok_emb + tgt_pe_emb
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

# --------------------------------------------------
# Test Seq2Seq Translation
# --------------------------------------------------
# seq2seq_translation = Seq2SeqTranslation(6, 6, 512, 8, 8500, 8000)
# src = torch.randint(0, 8500, (10, 32))
# tgt = torch.randint(0, 8000, (20, 32))
# out = seq2seq_translation(src, tgt, None, None, None, None, None)
# print(out.size())
# --------------------------------------------------