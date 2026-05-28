"""
获取 SentencePiece 分词器的脚本示例。
"""
from datasets import load_dataset
import sentencepiece as spm
import os
from typing import Tuple, List

from loguru import logger

PAD_IDX = 0    # 填充索引
UNK_IDX = 1    # 未知词索引
BOS_IDX = 2    # 句首索引
EOS_IDX = 3    # 句尾索引


# --- 1. Load dataset and preprocess ---
def prepare_data(df_cache_dir: str) -> Tuple:
    logger.info(f"Loading dataset... (cache dataset to {df_cache_dir})")
    # load_dataset 会自动下载并缓存数据集，cache_dir 参数指定了缓存目录
    raw_dataset = load_dataset("opus100", "en-zh", cache_dir=df_cache_dir)
    # 数据集包含 'train' 和 'validation' 两个 split，每个 split 中的每条数据包含 'translation' 字段，里面有 'en' 和 'zh' 两个子字段分别对应英文和中文句子
    train_texts = []
    val_texts = []
    for split_name in ['train', 'validation']:
        split_data = raw_dataset[split_name]
        texts = [(item['translation']['en'], item['translation']['zh']) for item in split_data]
        if split_name == 'train':
            train_texts.extend(texts)
        else:
            val_texts.extend(texts)
    return train_texts, val_texts


# --- 2. Train SentencePiece tokenizer ---
def train_sentencepiece(train_texts: List[Tuple[str, str]]) -> spm.SentencePieceProcessor:
    # 将训练集的中英句子拼接起来用于训练 SentencePiece 分词器
    all_text_for_sp = []
    for en_sent, zh_sent in train_texts:
        all_text_for_sp.append(en_sent)
        all_text_for_sp.append(zh_sent)

    # 训练 SentencePiece BPE 分词器
    sp_input_file = "spm_input.txt"
    with open(sp_input_file, 'w', encoding='utf-8') as f:
        for line in all_text_for_sp:
            f.write(line + "\n")

    sp_model_path = "bpe_model"
    if not os.path.exists(f"{sp_model_path}.model"):
        logger.info("Training SentencePiece model...")
        spm.SentencePieceTrainer.train(
            input=sp_input_file,
            model_prefix=sp_model_path,
            vocab_size=16000, # 词表大小
            character_coverage=1.0,     # 中文必须 1.0
            model_type='bpe',
            pad_id=PAD_IDX,
            unk_id=UNK_IDX,
            bos_id=BOS_IDX,
            eos_id=EOS_IDX,
        )
        os.remove(sp_input_file) # 清理临时文件

    sp = spm.SentencePieceProcessor()
    sp.load(f"{sp_model_path}.model")
    logger.info(f"Loaded SentencePiece model with vocab size: {sp.get_piece_size()}")
    return sp

if __name__ == "__main__":
    import math
    # hf dataset cache dir``
    cache_dir = "./hf_dataset_cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_texts, val_texts = prepare_data(cache_dir)
    sp = train_sentencepiece(train_texts)
    vocab_size = sp.get_piece_size()
    logger.info(f"Vocabulary size: {vocab_size}")
