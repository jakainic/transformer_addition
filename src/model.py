# -----------------------
# Build Model
# -----------------------
from transformers import GPT2Config, GPT2LMHeadModel

def build_model(vocab_size: int, n_ctx: int = 64, n_layer: int = 6, n_embd: int = 256, n_head: int = 8,
                pdrop: float = 0.1):
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_ctx,
        n_ctx=n_ctx,          # legacy GPT-2 field; keep same as n_positions
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=pdrop,
        embd_pdrop=pdrop,
        attn_pdrop=pdrop,
    )
    return GPT2LMHeadModel(cfg)
