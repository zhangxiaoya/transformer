import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# from torchtext.datasets import Multi30k
# Modify Multi30k in torchtext.datasets
from dataset.multi30k import Multi30k

from timeit import default_timer as timer

from model.seq2seq_translation import Seq2SeqTransformer
from dataset.data_util import DataUtil, DEVICE

# -- init data utility
data_util = DataUtil()
# -- use Multi30k dataset
train_iter = Multi30k(split='train', language_pair=(data_util.SRC_LANGUAGE, data_util.TGT_LANGUAGE))
val_iter = Multi30k(split='valid', language_pair=(data_util.SRC_LANGUAGE, data_util.TGT_LANGUAGE))
test_iter = Multi30k(split='test', language_pair=(data_util.SRC_LANGUAGE, data_util.TGT_LANGUAGE))
# -- init vocab and text transform
data_util.init_vocab(train_iter)
text_transform = data_util.init_text_transform()

# -- function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[data_util.SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[data_util.TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=data_util.PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=data_util.PAD_IDX)
    return src_batch, tgt_batch

# -- function to train the model for one epoch
def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = data_util.create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

# -- function to evaluate the model
def evaluate(model):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = data_util.create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


if __name__ == "__main__":
    torch.manual_seed(0)
    NUM_EPOCHS = 18
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    # -- init model
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, data_util.SRC_VOCAB_SIZE, data_util.TGT_VOCAB_SIZE, FFN_HID_DIM)
    # -- init weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # -- move model to device
    transformer = transformer.to(DEVICE)
    # -- init loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_util.PAD_IDX)
    # -- init optimizer
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    # -- train model
    os.makedirs("./checkpoints", exist_ok=True)
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = 0
        val_loss = evaluate(transformer)
        torch.save(transformer.state_dict(), f"./checkpoints/transformer_{epoch:03d}.pth")
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))