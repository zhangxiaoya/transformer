# SentencePiece（SP）从 0 到可用的完整实战说明

SentencePiece = 语言无关的子词分词器
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