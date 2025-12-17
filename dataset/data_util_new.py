import torch
import time
import logging
import jieba
from tqdm import tqdm
from datasets import load_dataset
from typing import Iterable, List, Generator, Optional, Any
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from constant.dataset import (
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    UNK_IDX,
    BOS_IDX,
    EOS_IDX,
)

for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

class DataUtil:
    def __init__(self,
                 src_field: Optional[str] = None,
                 tgt_field: Optional[str] = None,
                 src_tokenizer: Optional[Any] = None,
                 tgt_tokenizer: Optional[Any] = None):
        """
        Data utility for building tokenizers/vocabs and converting text -> tensor.
        Supports HuggingFace `datasets` examples (dicts) and simple (src,tgt) tuples.

        Args:
            src_field: optional column name for source text in dataset examples (e.g. 'zh' or 'source').
            tgt_field: optional column name for target text in dataset examples (e.g. 'en' or 'target').
            src_tokenizer / tgt_tokenizer: optional callable(text) -> List[str]. If None uses reasonable defaults.
        """
        # -- 自定义两个常量表示枚举
        self.language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
        # field names when dataset examples are dicts
        self.field_map = {SRC_LANGUAGE: src_field, TGT_LANGUAGE: tgt_field}
        # -- 定义两个全局变量，分别是token分词器 和 字典
        self.token_transform = {}
        self.vocab_transform = {}
        # default tokenizers: prefer provided, otherwise
        if src_tokenizer is not None:
            self.token_transform[SRC_LANGUAGE] = src_tokenizer
        else:
            # for Chinese use jieba if available, otherwise whitespace split
            self.token_transform[SRC_LANGUAGE] = lambda txt: list(jieba.cut(txt))

        if tgt_tokenizer is not None:
            self.token_transform[TGT_LANGUAGE] = tgt_tokenizer
        else:
            self.token_transform[TGT_LANGUAGE] = get_tokenizer('basic_english')

        # Make sure the tokens are in order of their indices to properly insert them in vocab
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

    # -- 定义一个helper函数，用来辅助生成token
    # helper function to yield list of tokens
    def yield_tokens(self, data_iter: Iterable, language: str) -> Generator[List[str], None, None]:
        """
        Accepts:
          - data_iter: an iterable of examples. Each example may be:
              * dict (huggingface dataset example) — tries 'translation' dict, or configured field_map, or keys (language)
              * tuple/list (src, tgt) — uses indices
              * raw string

        Yields token lists for building vocab.
        """
        for data_sample in tqdm(data_iter, desc=f"Building vocab tokens for {language}"):
            # resolve raw text depending on sample format
            text = None
            # HuggingFace dataset sample (dict)
            if isinstance(data_sample, dict):
                # case: {'translation': {'zh': ..., 'en': ...}}
                if 'translation' in data_sample and isinstance(data_sample['translation'], dict):
                    text = data_sample['translation'].get(language)
                # case: configured field names
                if text is None and self.field_map.get(language) is not None:
                    text = data_sample.get(self.field_map[language])
                # case: direct keys like 'zh' / 'en'
                if text is None:
                    text = data_sample.get(language)
                # last resort: pick first string-like value
                if text is None:
                    for v in data_sample.values():
                        if isinstance(v, str):
                            text = v
                            break
            elif isinstance(data_sample, (list, tuple)):
                try:
                    text = data_sample[self.language_index[language]]
                except Exception:
                    # fallback to first element
                    text = data_sample[0] if len(data_sample) > 0 else None
            else:
                text = data_sample

            if text is None:
                continue

            tokens = self.token_transform[language](text)
            # ensure tokens is iterable of str
            yield tokens

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))

    # -- 定义一个函数，用来初始化字典
    def init_vocab(self, train_iter):
        """
        Build vocabulary from an iterable dataset.

        train_iter can be:
         - a HuggingFace Dataset (dataset[*]) which is iterable
         - any iterable of examples (dicts, tuples, strings)

        Note: build_vocab_from_iterator consumes the generator, so we create a fresh generator per language.
        """
        # -- 训练数据集给源语言和目标语言生成字典
        start_t = time.time()
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            logger.info(f"Building vocab for language: {ln}")
            generator = (tokens for tokens in self.yield_tokens(train_iter, ln))
            self.vocab_transform[ln] = build_vocab_from_iterator(generator,
                                                                 min_freq=1,
                                                                 specials=self.special_symbols,
                                                                 special_first=True)

        # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            self.vocab_transform[ln].set_default_index(UNK_IDX)

        logger.info(f"Vocab building completed in {time.time() - start_t:.2f} seconds.")

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

if __name__ == "__main__":
    # Option 1: YogeLiu/zh-en-translation-dataset-600K
    # dataset_name = "YogeLiu/zh-en-translation-dataset-600K"
    # ds = load_dataset(
    #     dataset_name,
    #     cache_dir="/Users/ryancheung/workspace/hf_data_cache",
    #     revision="fdff0a91b2e8d756c8a4539d5f768c087979aa70",
    # )
    # src_field = "chinese"
    # tgt_field = "english"

    # Option 2: opus100 en-zh
    dataset_name = "opus100"
    ds = load_dataset(
        dataset_name,
        "en-zh",
        split="train",
        cache_dir="/Users/ryancheung/workspace/hf_data_cache",
        revision="805090dc28bf78897da9641cdf08b61287580df9",
    )
    src_field = "zh"
    tgt_field = "en"

    data_util = DataUtil(src_field=src_field, tgt_field=tgt_field)
    # 传入 ds['train']（可迭代）初始化 vocab
    data_util.init_vocab(ds['translation'])

    text_transform = data_util.init_text_transform()
    # 将一条样例转换为 tensor:
    for i in range(5):
        example = ds['translation'][i]
        src_tensor = text_transform[SRC_LANGUAGE](example['zh'])
        tgt_tensor = text_transform[TGT_LANGUAGE](example['en'])
        logger.info(f"Example {i}:")
        logger.info(f"  Source text: {example['zh']}")
        logger.info(f"  Source tensor: {src_tensor}")
        logger.info(f"  Target text: {example['en']}")
        logger.info(f"  Target tensor: {tgt_tensor}")
