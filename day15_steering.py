"""
Day 15 — Steering: turn a feature up and watch the output bend.

Each feature is a DIRECTION in activation space (its decoder row). To steer, we
add that direction — scaled by a strength alpha — into the residual stream at
layer 6 during generation. alpha=0 is normal GPT-2; crank alpha up and the
model's text bends toward that concept. This is the causal proof a feature is real.
"""

import torch
from transformer_lens import HookedTransformer
from sae import SparseAutoencoder

torch.set_grad_enabled(False)
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = HookedTransformer.from_pretrained("gpt2", device=device)
ckpt = torch.load("activations/gpt2_sae.pt")
sae = SparseAutoencoder(768, ckpt["n_features"]).to(device)
sae.load_state_dict(ckpt["state_dict"])
scale = ckpt["scale"].to(device)
LAYER = ckpt["layer"]
HOOK = f"blocks.{LAYER}.hook_resid_post"

FEATURE = 1225          # the patent "invention" feature
PROMPT = "My favorite thing to do on the weekend is"
ALPHAS = [0, 12, 24, 48]

# Feature direction in REAL activation space = decoder row * scale
# (the SAE trained on normalized acts, so a unit step there = `scale` in real space).
direction = sae.W_dec[FEATURE] * scale       # [768]

def steer(vec):
    def hook(value, hook):
        return value + vec
    return hook

print(f"Steering feature {FEATURE} (patent 'invention'). Prompt: {PROMPT!r}\n")
for alpha in ALPHAS:
    torch.manual_seed(0)                      # same randomness each run -> only steering differs
    vec = alpha * direction
    with model.hooks(fwd_hooks=[(HOOK, steer(vec))]):
        out = model.generate(PROMPT, max_new_tokens=40, temperature=0.7, verbose=False)
    tag = "(normal GPT-2)" if alpha == 0 else f"(steered x{alpha})"
    print(f"alpha={alpha:>3} {tag}:\n  {out}\n")
