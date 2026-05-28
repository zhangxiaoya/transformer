# 数据准备

## 1. 数据集下载

在GitHub上一些介绍Transformer的代码比较早起，使用Multi30k 数据集，这个数据集比较老旧且数据量偏小，现在主要用于教学演示，更新版本的torch一节分词工具都不支持，如果你想用更主流、更能验证 Transformer 模型有效性的机器翻译数据集和分词工具，推荐你使用 Hugging Face 的 datasets 和 tokenizers 库。

### Hugging Face 生态（datasets + tokenizers）
Hugging Face 的 tokenizers 库底层由 Rust 编写，速度极快，并且完美支持 Transformer 模型常用的 BPE (Byte Pair Encoding) 和 WordPiece 等子词分词算法。

1. 获取主流翻译数据集（以 WMT14 英德翻译为例）：
不再需要手动下载和解析老旧数据集，一行代码即可加载：
```python
from datasets import load_dataset

# 加载 WMT14 英德翻译数据集
dataset = load_dataset("wmt14", "de-en")
print(dataset["train"])
# 输出示例: {'translation': {'de': '...', 'en': '...'}}
```

2. 训练或使用现代分词器（Tokenizer）：
Transformer 模型通常使用“子词分词（Subword Tokenization）”来解决未登录词和词表过大的问题。你可以直接使用预训练的分词器（如 transformers 库自带的），也可以在自己的数据上训练一个 BPE 分词器：
```python
from transformers import AutoTokenizer

# 方法A：直接使用官方预训练的分词器（推荐，如 mBART, T5 等支持多语言翻译的模型）
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

# 方法B：如果手写 Transformer，可以用 tokenizers 库自己训练 BPE
# from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
# tokenizer = Tokenizer(models.BPE())
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# ...（配置训练器并在数据集上训练）
```

### OPUS-100 数据集
OPUS-100 / TED Talks：包含大量口语化的演讲翻译，句子相对简短。非常适合小规模快速验证。
```python
from datasets import load_dataset
# 加载 OPUS-100 中英平行语料
dataset = load_dataset("opus100", "en-zh")
```
- 下面是一个完整的例子
> data/load_dataset.py
```python
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

```



## 2.  SentencePiece（SP）从 0 到可用的完整实战说明

SentencePiece: 语言无关的子词分词器
不需要分词规则，中文/英文/混合文本都能直接用

1. 安装
```
pip install sentencepiece
```

或者
```
pip install sentencepiece -i https://pypi.org/simple
```

2. 准备语料数据

中英混合训练用一个 tokenizer，同一个词表，模型更稳定

```Python
import logging
from time import time
from tqdm import tqdm
from datasets import load_dataset

for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


start_t = time()
ds = load_dataset("opus100", "en-zh")

with open("opus100.txt", "w", encoding="utf-8") as f:
    for ex in tqdm(ds["train"], desc="Writing examples"):
        f.write(ex["translation"]["en"] + "\n")
        f.write(ex["translation"]["zh"] + "\n")

logger.info(f"Data writing completed in {time() - start_t:.2f} seconds.")
```

3. 训练SentencePiece模型
```Python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="opus100.txt",
    model_prefix="spm",
    vocab_size=16000,
    model_type="bpe",           # 推荐
    character_coverage=1.0,     # 中文必须 1.0
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)
```

完成后回生成这两个文件
```Bash
spm.model
spm.vocab
```

4. 加载分词模型并使用

```Python
sp = spm.SentencePieceProcessor()
sp.load("spm.model")
```
- 分词
```
text = "我喜欢 Natural Language Processing"
tokens = sp.encode(text, out_type=str)
print(tokens)
```

- 输出
['▁我', '▁喜欢', '▁Natural', '▁Language', '▁Processing']


**▁** 表示“词起始”

- 文本 → ID（最常用）
```Python
ids = sp.encode(text, out_type=int)
print(ids)
```

- 加 BOS / EOS（翻译必备）
```
BOS = sp.bos_id()
EOS = sp.eos_id()

ids = [BOS] + ids + [EOS]
```

- Padding（Tensor 化前一步）
```Python
def pad(ids, max_len=50):
    ids = ids[:max_len]
    return ids + [sp.pad_id()] * (max_len - len(ids))
```

5. 完整：文本 → Tensor
```Python
import torch

def text_to_tensor(text, max_len=50):
    ids = [sp.bos_id()] + sp.encode(text, out_type=int) + [sp.eos_id()]
    ids = ids[:max_len]
    ids += [sp.pad_id()] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

x = text_to_tensor("我喜欢 NLP")
print(x.shape)  # [50]
```

6. 反向：ID → 文本（推理必用）
```Python
ids = x.tolist()
text = sp.decode(ids)
print(text)
```


- 会自动忽略 <pad>

7. 在 Transformer 训练中如何用（关键点）
```Python
# Encoder 输入（source）
src = text_to_tensor(en_text)

# Decoder 输入（target input）
tgt_input = text_to_tensor(zh_text)[:-1]
tgt_output = text_to_tensor(zh_text)[1:]
```