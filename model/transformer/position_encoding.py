import torch.nn as nn
import torch

class PositionEncodings(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEncodings, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        import pdb; pdb.set_trace()
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        import pdb; pdb.set_trace()
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# -- test code
import matplotlib.pyplot as plt
pe = PositionEncodings(512, 0)
y = pe(torch.zeros(1, 100, 512))
plt.figure(figsize=(15, 5))
plt.plot(y[0, :, 4:8].data.numpy())
plt.legend(['dim %d' % p for p in [4, 5, 6, 7]])
plt.savefig('position_encodings.png')
