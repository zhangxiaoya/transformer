import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(sefl, d_model, vocab_size):
        super(Embeddings, self)._init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)