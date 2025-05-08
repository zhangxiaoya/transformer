
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super(TokenEmbedding, self).__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dim)

    def forward(self, x):
        return self.embedding(x.long()) * torch.sqrt(torch.tensor(self.model_dim))

# --------------------------------------------------
# Test Token Embedding
# --------------------------------------------------
# token_emb = TokenEmbedding(1000, 512)
# test_input = torch.randint(0, 1000, (128, 50))
# output = token_emb(test_input)
# print(output.size())


class PositionEncoding(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super(PositionEncoding, self).__init__()
        self.model_size = model_dim
        self.vocab_size = vocab_size
        position_encoding_weight = torch.zeros(vocab_size, model_dim)
        _2i = torch.arange(0, model_dim, 2)
        pos = torch.arange(0, vocab_size).unsqueeze(1)
        position_encoding_weight[:, 0::2] = torch.sin(pos / (vocab_size ** (_2i / model_dim)))
        position_encoding_weight[:, 1::2] = torch.cos(pos / (vocab_size ** (_2i / model_dim)))
        self.encoding = nn.Embedding.from_pretrained(position_encoding_weight, freeze=True)

    def forward(self, x):
        return self.encoding(x.long())

# --------------------------------------------------
# Test Position Encoding
# --------------------------------------------------
# position_encoding = PositionEncoding(1000, 512)
# test_input = torch.randint(0, 1000, (128, 50))
# output = position_encoding(test_input)
# print(output.size())

