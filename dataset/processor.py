import torch
from constant.train import DEFAULT_DEVICE
from constant.dataset import PAD_IDX

def generate_square_subsequent_mask(
    sz: int,
    device: torch.device = DEFAULT_DEVICE
) -> torch.Tensor:
    """Generates a square mask for the sequence.

    Args:
        sz (int): sequence length
        device (torch.device, optional): device id. Defaults to DEFAULT_DEVICE.

    Returns:
        torch.Tensor: generated mask tensor.
    """
    # Generate down trangular(value as 1) mask:
    #   1. Generates a square matrix of size sz x sz with all values as 1;
    #   2. Using triu to returns the upper triangular part of a matrix (2-D tensor);
    #   3. transpose to down triangular part by swapping the rows and columns;
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    #   1. Convert the mask to float and set the masked positions to -inf and unmasked positions to 0
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(
    src: torch.Tensor,
    tgt: torch.Tensor,
    device: torch.device = DEFAULT_DEVICE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates masks for the source and target sequences.

    Args:
        src (torch.Tensor): src sequence tensor
        tgt (torch.Tensor): tgt sequence tensor
        device (torch.device, optional): Device. Defaults to DEFAULT_DEVICE.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: mask for src, tgt, src_padding_mask, tgt_padding_mask
    """
    src_seq_len: int = src.shape[0]
    tgt_seq_len: int = tgt.shape[0]
    # For target squences, we need to create a mask that prevents the model from attending to future tokens.
    tgt_mask: torch.Tensor = generate_square_subsequent_mask(tgt_seq_len, device)
    # For source sequences, we not need to create a mask, so we can use a zero matrix.
    src_mask: torch.Tensor = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    # Create padding masks for source and target sequences
    src_padding_mask: torch.Tensor = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask: torch.Tensor = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask