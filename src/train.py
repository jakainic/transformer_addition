from model import build_model
from tokenizer import CharacterTokenizer

import os
import numpy as np
import random
import torch
from transformers import Trainer, TrainingArguments

# -----------------------
# Train
# -----------------------
def seed_everything(seed: int = 0, deterministic: bool = False):
    """
    Seeds Python, NumPy, and Torch. If deterministic=True, also enables
    (slower) deterministic CUDA behavior where possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Slower but more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # If you're on PyTorch >= 1.8, this enforces more determinism:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def main():
    seed_everything(42)
    tok = CharacterTokenizer()
    model = build_model(len(tok.vocab))
    train_ds = AddDataset(n=200_000, k=3, mode="uniform", max_len=64)
    val_ds = AddDataset(n=10_000,  k=3, mode="uniform", max_len=64)

    args = TrainingArguments(
      output_dir="add_gpt",
      per_device_train_batch_size=256,
      per_device_eval_batch_size=256,
      gradient_accumulation_steps=1,
      learning_rate=3e-4,
      warmup_steps=200,
      max_steps=8000,
      fp16=True,
      logging_steps=100,
      save_steps=1000,
      save_total_limit=2,
      report_to="none",
    )
  
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    trainer.save_model("checkpoints/final")
