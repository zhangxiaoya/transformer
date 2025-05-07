from dataset.data_util import DataUtil
from dataset.multi30k import Multi30k
from torch.nn.utils.rnn import pad_sequence

from constant.dataset import (
    DATASET_CACHE_DIR,
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    TRAIN,
    PAD_IDX,
)

# Init vocabilary using Train dataset
language_pair = (SRC_LANGUAGE, TGT_LANGUAGE)
train_iter = Multi30k(
    root=DATASET_CACHE_DIR,
    split=TRAIN,
    language_pair=language_pair,
)
data_util = DataUtil()
data_util.init_vocab(train_iter)

# Init text transform which transforms text to tensor, including:
# 1. Tokenization
# 2. Numericalization (Vocabulary lookup)
# 3. Tensor conversion, added <BOS> and <EOS> tokens
text_transform = data_util.init_text_transform()

def collate_fn(batch):
    src_batch = []
    tgt_bacth = []
    for src_sample, tgt_sampl in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_bacth.append(text_transform[TGT_LANGUAGE](tgt_sampl.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_bacth = pad_sequence(tgt_bacth, padding_value=PAD_IDX)
    return src_batch, tgt_bacth
