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

