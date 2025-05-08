import logging

# third-party libraries
from torchtext.data.utils import get_tokenizer

# custom libraries
from dataset.multi30k import Multi30k
from constant.dataset import (
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    DATASET_CACHE_DIR,
)

en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
de_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")

for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

multi_30k_train_iter = Multi30k(
    root=DATASET_CACHE_DIR,
    split="train",
    language_pair=(SRC_LANGUAGE, TGT_LANGUAGE),
)
for i, (src_data, tgt_data) in enumerate(multi_30k_train_iter):
    logger.info(f"Processing batch {i}")
    src_seq = src_data
    tgt_seq = tgt_data
    logger.info(f"Source input: {src_seq}")
    logger.info(f"Target input: {tgt_seq}")
    src_seq = de_tokenizer(src_seq)
    tgt_seq = en_tokenizer(tgt_seq)
    logger.info(f"Source tokenized: {src_seq}")
    logger.info(f"Target tokenized: {tgt_seq}")
    if i == 1:
        break