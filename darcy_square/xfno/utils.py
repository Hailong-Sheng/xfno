import yaml
import torch
import numpy as np
from easydict import EasyDict as edict

def load_config(path):
    with open(path, 'r') as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    return config

def set_torch_dtype(dtype):
    if dtype=='float16':
        torch_dtype = torch.float16
    if dtype=='float32':
        torch_dtype = torch.float32
    if dtype=='float64':
        torch_dtype = torch.float64
    torch.set_default_dtype(torch_dtype)
    return torch_dtype

def set_random_seed(seed: int=0):
    torch.manual_seed(seed)
    np.random.seed(seed)