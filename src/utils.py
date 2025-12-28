# ---------
# Helpers
# ---------
import os, random
import numpy as np
import torch

def seed_everything(seed: int = 0, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def make_rng(seed: int):
    """Local RNG (don’t mutate global random state)."""
    return random.Random(seed)
