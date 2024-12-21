# Basic data process for NLP (using Multi30 dataset), include tokenizer, vocab, and dataloader.
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 1. Define global constants
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
Language_index = {
    SRC_LANGUAGE: 0,
    TGT_LANGUAGE: 1,
}
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
special_sysbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# 2. Instantiate dataset iterators
train_iter = Multi30k(split='train', language_pair= (SRC_LANGUAGE, TGT_LANGUAGE))
test_iter = Multi30k(split='test', language_pair= (SRC_LANGUAGE, TGT_LANGUAGE))
val_iter = Multi30k(split='valid', language_pair= (SRC_LANGUAGE, TGT_LANGUAGE))
# -------------------------------------
# print some data sample
# -------------------------------------
print("--> Print some data sample:")
for idx, data_sample in enumerate(train_iter):
    if idx > 5:
        break
    print(data_sample[Language_index[SRC_LANGUAGE]])
    print(data_sample[Language_index[TGT_LANGUAGE]])
    print("")
# -------------------------------------

#  3. Tokenizer
src_tokenizer = get_tokenizer('spacy', language = 'de_core_news_sm')
tgt_tokenizer = get_tokenizer('spacy', language = 'en_core_web_sm')

tokenizers = {
    SRC_LANGUAGE: src_tokenizer,
    TGT_LANGUAGE: tgt_tokenizer
}
# -------------------------------------
# print some data sample
# -------------------------------------
print("--> Print some data sample tokenized result:")
for idx, data_sample in enumerate(train_iter):
    if idx > 5:
        break
    print("src", tokenizers[SRC_LANGUAGE](data_sample[Language_index[SRC_LANGUAGE]]))
    print("tgt", tokenizers[TGT_LANGUAGE](data_sample[Language_index[TGT_LANGUAGE]]))
    print("")

# 4.Build Vocabs
def yield_tokens(data_iter, language):
    for data_sample in data_iter:
        yield tokenizers[language](data_sample[Language_index[language]])

vocab_transforms = {}
for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transforms[language] = build_vocab_from_iterator(
        yield_tokens(train_iter, language),
        min_freq=1,
        specials=special_sysbols,
        special_first=True
    )
# -------------------------------------
# print some data sample after vocab
# -------------------------------------
print("--> Print some data sample tokenized result after vocab:")
for idx, data_sample in enumerate(train_iter):
    if idx > 5:
        break
    src_data = data_sample[Language_index[SRC_LANGUAGE]]
    print("SRC", src_data)
    print("tokenized:", tokenizers[SRC_LANGUAGE](src_data))
    print("index:", vocab_transforms[SRC_LANGUAGE](tokenizers[SRC_LANGUAGE](src_data)))
    tgt_data = data_sample[Language_index[TGT_LANGUAGE]]
    print("TGT", tgt_data)
    print("tokenized:", tokenizers[TGT_LANGUAGE](tgt_data))
    print("index:", vocab_transforms[TGT_LANGUAGE](tokenizers[TGT_LANGUAGE](tgt_data)))
    print("")
# -------------------------------------

# 5. Multi transform process, include tokenization, numericalization, and tensorization
def tensor_transform(token_ids):
    return torch.cat((
        torch.tensor([BOS_IDX]),
        torch.tensor(token_ids),
        torch.tensor([BOS_IDX]),
    ))

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

data_transforms = {}
for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
    data_transforms[language] = sequential_transforms(
        tokenizers[language],
        vocab_transforms[language],
        tensor_transform,
    )
# -------------------------------------
# print some data sample after multi transform
# -------------------------------------
print("--> Print some data sample tokenized result after multi transform:")
for idx, data_sample in enumerate(train_iter):
    if idx > 5:
        break
    src_data = data_sample[Language_index[SRC_LANGUAGE]]
    print("SRC", src_data)
    print("tokenized:", tokenizers[SRC_LANGUAGE](src_data))
    print("index:", vocab_transforms[SRC_LANGUAGE](tokenizers[SRC_LANGUAGE](src_data)))
    print("tensor:", data_transforms[SRC_LANGUAGE](src_data))
    tgt_data = data_sample[Language_index[TGT_LANGUAGE]]
    print("TGT", tgt_data)
    print("tokenized:", tokenizers[TGT_LANGUAGE](tgt_data))
    print("index:", vocab_transforms[TGT_LANGUAGE](tokenizers[TGT_LANGUAGE](tgt_data)))
    print("tensor:", data_transforms[TGT_LANGUAGE](tgt_data))
    print("")
# -------------------------------------

# 6. Define dataloader
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_data, tgt_data in batch:
        src_batch.append(data_transforms[SRC_LANGUAGE](src_data))
        tgt_batch.append(data_transforms[TGT_LANGUAGE](tgt_data))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

BATCH_SIZE = 128
train_dataloader = DataLoader(train_iter, batch_size = BATCH_SIZE, collate_fn = collate_fn)
test_dataloader = DataLoader(test_iter, batch_size = BATCH_SIZE, collate_fn = collate_fn)
val_dataloader = DataLoader(val_iter, batch_size = BATCH_SIZE, collate_fn = collate_fn)

# -------------------------------------
# print some data sample after dataloader
# -------------------------------------
print("--> Print some data sample tokenized result after dataloader:")
for idx, (src_batch, tgt_batch) in enumerate(train_dataloader):
    if idx > 5:
        break
    print("src_batch", src_batch.shape)
    print("tgt_batch", tgt_batch.shape)
    print("")
# -------------------------------------