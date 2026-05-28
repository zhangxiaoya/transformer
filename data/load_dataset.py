"""
从 Hugging Face 下载数据集并缓存到本地的脚本示例。
这个脚本使用 Hugging Face 的 `datasets` 库来加载数据集，并将数据集缓存到指定的本地目录。它还包含了训练 SentencePiece 分词器的代码示例。
"""
import logging
import os
from typing import Tuple
from time import time
from datasets import load_dataset

for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# 在没有网络的环境，需要把这两个环境变量打开
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 从HF下载数据集配置
DATASET_NAME = "opus100"
DATASET_CONFIG = "en-zh"


# --- 1. Load dataset and preprocess ---
def prepare_data(df_cache_dir: str) -> Tuple:
    startT = time()
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
    logger.info(f"Dataset loaded and preprocessed in {time() - startT:.2f} seconds.")
    return train_texts, val_texts

if __name__ == "__main__":
    # hf dataset cache dir``
    cache_dir = "./hf_dataset_cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_texts, val_texts = prepare_data(cache_dir)
