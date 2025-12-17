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