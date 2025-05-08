import torch.nn as nn
import copy
import numpy as np
import torch

def clones(module, N):
    """Produce N identical layers.

    Args:
        module (nn.Module): Module to be copied
        N (int): Number of copies
    """
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))
    # return nn.ModuleList([module() for _ in range(N)])

def subsequent_mask(size):
    """Mask out subsequent positions.

    Args:
        size (int): Size of mask
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # Upper triangular matrix
    return torch.from_numpy(subsequent_mask) == 0 # Lower triangular matrix

# -- test code
# import matplotlib.pyplot as plt
# plt.figure(figsize=(5,5))
# plt.imshow(subsequent_mask(20)[0])
# plt.savefig('subsequent_mask.png')
