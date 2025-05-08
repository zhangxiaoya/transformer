# system libraries
import logging

# third-party libraries
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# custom libraries
from dataset.multi30k import Multi30k
from constant.dataset import (
    SRC_LANGUAGE,
    TGT_LANGUAGE,
    DATASET_CACHE_DIR,
)
from dataset.collate_fn import collate_fn
from dataset.processor import (
    create_mask,
)

for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    SRC_LANGUAGE = "de"
    TGT_LANGUAGE = "en"
    multi_30k_train_iter = Multi30k(
        root=DATASET_CACHE_DIR,
        split="train",
        language_pair=(SRC_LANGUAGE, TGT_LANGUAGE),
    )

    multi_30k_train_loader = DataLoader(
        multi_30k_train_iter,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn,
    )
    for i, (src_data, tgt_data) in enumerate(multi_30k_train_loader):
        logger.info(f"Batch {i}: {src_data.shape}, {tgt_data.shape}")

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_data, tgt_data)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(src_mask.cpu().numpy())
        plt.title("Source Mask")
        plt.subplot(1, 2, 2)
        plt.imshow(tgt_mask.cpu().numpy())
        plt.title("Target Mask")
        plt.savefig(f"data mask {i}.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(src_padding_mask.cpu().numpy())
        plt.title("Source Padding Mask")
        plt.subplot(1, 2, 2)
        plt.imshow(tgt_padding_mask.cpu().numpy())
        plt.title("Target Padding Mask")
        plt.savefig(f"padding mask {i}.png")
        plt.close()
        if i == 1:
            break