import torch

# default device
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
