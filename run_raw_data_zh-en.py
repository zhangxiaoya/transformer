# This code snippet used to test dataset loading,
import logging
from datasets import load_dataset


for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    dataset_cache_dir = "/Users/ryancheung/workspace/dataset_cache"
    SRC_LANGUAGE = "zh"
    TGT_LANGUAGE = "en"
    # multi_30k_train_iter = Multi30k(
    #     root=dataset_cache_dir,
    #     split="train",
    #     language_pair=(SRC_LANGUAGE, TGT_LANGUAGE),
    # )
    # # logging out raw data
    # for i, (src_data, tgt_data) in enumerate(multi_30k_train_iter):
    #     logger.info(f"Batch {i}: {src_data}, {tgt_data}")
    #     # if i == 5:
    #     #     break

    # ds = load_dataset("ted2020", "zh-en")
    # from datasets import load_dataset

    # Login using e.g. `huggingface-cli login` to access this dataset
    # ds = load_dataset("bigscience-data/roots_en_ted_talks_iwslt")
    # from datasets import load_dataset

    # Login using e.g. `huggingface-cli login` to access this dataset
    # ds = load_dataset("opus_books", "zh-en")
    # ds = load_dataset("iwslt2017", "zh-en")
    # ds = load_dataset("opus100", "zh-en")
    ds = load_dataset("opus100", "en-zh")
    # ds = load_dataset("ted2020", "zh-en")
    # ds = load_dataset("YogeLiu/zh-en-translation-dataset-600K")
    for idx, example in enumerate(ds['train']):
        logger.info(f"Example {idx}: {example}")
        if idx == 5:
            break

    # iwslt_train_iter = IWSLT2017(
    #     root=dataset_cache_dir,
    #     split="train",
    #     language_pair=(SRC_LANGUAGE, TGT_LANGUAGE),
    # )

    # for i, (src_data, tgt_data) in enumerate(iwslt_train_iter):
    #     logger.info(f"Batch {i}: {src_data}, {tgt_data}")
    #     if i == 5:
    #         break


    # multi_30k_train_iter = Multi30k(
    #     root=dataset_cache_dir,
    #     split="test",
    #     language_pair=(SRC_LANGUAGE, TGT_LANGUAGE),
    # )
    # # logging out raw data
    # for i, (src_data, tgt_data) in enumerate(multi_30k_train_iter):
    #     logger.info(f"Batch {i}: {src_data}, {tgt_data}")
    #     # if i == 5:
    #     #     break
