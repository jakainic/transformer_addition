# -------------
# Train Model
# -------------

from src.data import AdditionDataset
from src.model import build_model
from src.tokenizer import CharacterTokenizer
from src.utils import seed_everything

import argparse
import json
from pathlib import Path
from transformers import Trainer, TrainingArguments

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="runs/add_k3")
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    # model size knobs
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_head", type=int, default=8)

    # train knobs
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=8000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--train_bs", type=int, default=64)
    p.add_argument("--eval_bs", type=int, default=128)
    p.add_argument("--grad_accum", type=int, default=1)

    args = p.parse_args()
    seed_everything(args.seed)

    tok = CharacterTokenizer()
    train_ds = AdditionDataset(args.train_path, tok, max_len=args.max_len)
    val_ds   = AdditionDataset(args.val_path, tok, max_len=args.max_len)

    model = build_model(
    vocab_size=len(tok.vocab),
    n_ctx=args.max_len,
    n_layer=args.n_layer,
    n_embd=args.n_embd,
    n_head=args.n_head,
    )

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        fp16=True,
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        eval_strategy="steps",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )
  
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    trainer.save_model(args.out_dir)

    log_path = Path(args.out_dir) / "log_history.json"
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print("Saved logs to", log_path)

if __name__ == "__main__":
    main()
