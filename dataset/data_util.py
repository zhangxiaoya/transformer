
import torch
from typing import Iterable, List
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# default device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

class DataUtil:
    def __init__(self):
        # -- 自定义两个常量表示枚举
        self.SRC_LANGUAGE = 'de'
        self.TGT_LANGUAGE = 'en'
        self.language_index = {self.SRC_LANGUAGE: 0, self.TGT_LANGUAGE: 1}
        # -- 定义两个全局变量，分别是token分词器 和 字典
        self.token_transform = {}
        self.vocab_transform = {}
        # -- 用刚刚定义的全局变量token_transform 记录源语言和目标语言分词器；
        self.token_transform[self.SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
        self.token_transform[self.TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
        # 定义一些特殊字符的index
        # Define special symbols and indices
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    # -- 定义一个helper函数，用来辅助生成token
    # helper function to yield list of tokens
    def yield_tokens(self, data_iter: Iterable, language: str) -> List[str]:
        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[self.language_index[language]])

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([self.EOS_IDX])))

    # -- 定义一个函数，用来初始化字典
    def init_vocab(self, train_iter):
        # -- 训练数据集给源语言和目标语言生成字典
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            # Create torchtext's Vocab object
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln),
                                                                 min_freq=1,
                                                                 specials=self.special_symbols,
                                                                 special_first=True)

        # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
        # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            self.vocab_transform[ln].set_default_index(self.UNK_IDX)

    @property
    def SRC_VOCAB_SIZE(self):
        return len(self.vocab_transform[self.SRC_LANGUAGE])

    @property
    def TGT_VOCAB_SIZE(self):
        return len(self.vocab_transform[self.TGT_LANGUAGE])

    def init_text_transform(self):
        # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
        text_transform = {}
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            text_transform[ln] = sequential_transforms(self.token_transform[ln], #Tokenization
                                                       self.vocab_transform[ln], #Numericalization
                                                       self.tensor_transform) # Add BOS/EOS and create tensor
        return text_transform

    def generate_square_subsequent_mask(self, sz, device=DEVICE):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt, device=DEVICE):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

        src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask