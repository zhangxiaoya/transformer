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