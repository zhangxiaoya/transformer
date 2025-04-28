import torch
from typing import Iterable, List, Generator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from constant.dataset import (
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    UNK_IDX,
    BOS_IDX,
    EOS_IDX,
)

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
        self.language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
        # -- 定义两个全局变量，分别是token分词器 和 字典
        self.token_transform = {}
        self.vocab_transform = {}
        # -- 用刚刚定义的全局变量token_transform 记录源语言和目标语言分词器；
        self.token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
        self.token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    # -- 定义一个helper函数，用来辅助生成token
    # helper function to yield list of tokens
    def yield_tokens(self, data_iter: Iterable, language: str) -> Generator[str, None, None]:
        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[self.language_index[language]])

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))

    # -- 定义一个函数，用来初始化字典
    def init_vocab(self, train_iter):
        # -- 训练数据集给源语言和目标语言生成字典
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            # Create torchtext's Vocab object
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln),
                                                                 min_freq=1,
                                                                 specials=self.special_symbols,
                                                                 special_first=True)

        # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
        # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            self.vocab_transform[ln].set_default_index(UNK_IDX)

    @property
    def SRC_VOCAB_SIZE(self):
        return len(self.vocab_transform[SRC_LANGUAGE])

    @property
    def TGT_VOCAB_SIZE(self):
        return len(self.vocab_transform[TGT_LANGUAGE])

    def init_text_transform(self):
        # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
        text_transform = {}
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            text_transform[ln] = sequential_transforms(self.token_transform[ln], #Tokenization
                                                       self.vocab_transform[ln], #Numericalization
                                                       self.tensor_transform) # Add BOS/EOS and create tensor
        return text_transform
