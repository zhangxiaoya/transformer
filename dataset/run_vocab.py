# system libraries
import logging

# third-party libraries
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

# custom libraries
from dataset.multi30k import Multi30k
from constant.dataset import (
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    DATASET_CACHE_DIR,
    UNK_IDX,
)
for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

dataset = Multi30k(
    root=DATASET_CACHE_DIR,
    split="train",
    language_pair=(SRC_LANGUAGE, TGT_LANGUAGE),
)

language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

def yeild_dataset(dataset, language):
    for data_sample in dataset:
        yield data_sample[language_index[language]]

if __name__ == "__main__":
    # -- 生成源语言和目标语言的字典
    vocab_transform = {}
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yeild_dataset(dataset, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        dataset.reset()
        vocab_transform[ln].set_default_index(UNK_IDX)

    tokenizer_transform = {}
    tokenizer_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
    tokenizer_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
    for i, (src_data, tgt_data) in enumerate(dataset):
        logger.info(f"Processing batch {i}")
        src_seq = src_data
        tgt_seq = tgt_data
        logger.info(f"Source input: {src_seq}")
        logger.info(f"Target input: {tgt_seq}")
        src_seq = tokenizer_transform[SRC_LANGUAGE](src_seq)
        tgt_seq = tokenizer_transform[TGT_LANGUAGE](tgt_seq)
        logger.info(f"Source tokenized: {src_seq}")
        logger.info(f"Target tokenized: {tgt_seq}")
        src_seq = vocab_transform[SRC_LANGUAGE](src_seq)
        tgt_seq = vocab_transform[TGT_LANGUAGE](tgt_seq)
        logger.info(f"Source vocab: {src_seq}")
        logger.info(f"Target vocab: {tgt_seq}")
        if i == 1:
            break