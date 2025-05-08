import torch
import torch.nn as nn

transformer_model = nn.Transformer(
    nhead = 16,
    num_decoder_layers=12,
    num_encoder_layers=12,
    d_model=512,
    dim_feedforward=2048,
    dropout=0.1
)

src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)
print(out.size())
# print(out)