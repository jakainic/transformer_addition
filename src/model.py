# -----------------------
# Build Model
# -----------------------
from transformers import GPT2Config, GPT2LMHeadModel

def build_model(vocab_size, n_ctx=64):
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_ctx,
        n_ctx=n_ctx,
        n_embd=256,
        n_layer=6,
        n_head=8,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    return GPT2LMHeadModel(cfg)
