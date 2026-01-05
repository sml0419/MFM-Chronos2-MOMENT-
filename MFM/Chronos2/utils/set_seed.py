import numpy as np
import torch
import os
import random

def set_seed(seed):
    fix_seed = seed
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    random.seed(fix_seed)