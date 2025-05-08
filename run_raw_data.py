# This code snippet used to test dataset loading,
import logging
from dataset.multi30k import Multi30k

for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    dataset_cache_dir = "/workspaces/dataset_cache"
    SRC_LANGUAGE = "de"
    TGT_LANGUAGE = "en"
    multi_30k_train_iter = Multi30k(
        root=dataset_cache_dir,
        split="train",
        language_pair=(SRC_LANGUAGE, TGT_LANGUAGE),
    )
    # logging out raw data
    for i, (src_data, tgt_data) in enumerate(multi_30k_train_iter):
        logger.info(f"Batch {i}: {src_data}, {tgt_data}")
        if i == 5:
            break
