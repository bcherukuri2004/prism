"""
Day 11 — Read the features: what text makes each one fire?

We have a trained SAE with ~2,000 live features, but they're just numbered.
To learn what a feature MEANS, we find its "max-activating examples": scan all
our tokens and show the ones that light that feature up the most. A feature that
fires on { 1987, 1991, 2003, 1962 } is a "years" feature, and so on.
"""

import torch
from transformers import AutoTokenizer
from sae import SparseAutoencoder

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Load the trained SAE + its normalization stats --------------------------
ckpt = torch.load("activations/gpt2_sae.pt")
sae = SparseAutoencoder(d_model=768, n_features=ckpt["n_features"]).to(device)
sae.load_state_dict(ckpt["state_dict"])
mean, scale = ckpt["mean"].to(device), ckpt["scale"].to(device)

# --- Load the activations + the tokens they came from ------------------------
d = torch.load("activations/layer6_resid.pt")
acts = d["activations"]                 # [32000, 768]
tokens = d["tokens"]                     # [250, 128]  (token ids per document)
SEQ_LEN = tokens.shape[1]
tok = AutoTokenizer.from_pretrained("gpt2")

# --- Run every activation through the SAE encoder -> feature firings ----------
print("Computing feature activations for all 32,000 tokens...")
feats = []
with torch.no_grad():
    for i in range(0, acts.shape[0], 4096):
        x = ((acts[i:i + 4096].to(device) - mean) / scale)
        feats.append(sae.encode(x).cpu())
F = torch.cat(feats)                     # [32000, 2048]

# Ignore position 0 of each doc (the <|endoftext|> marker fires oddly).
valid = torch.tensor([r % SEQ_LEN != 0 for r in range(F.shape[0])])
Fv = F.clone()
Fv[~valid] = 0.0

# --- Pick a few SPECIFIC features (fire rarely + strongly = interpretable) -----
freq = (Fv > 0).float().mean(dim=0)           # how often each feature fires
peak = Fv.max(dim=0).values                   # how hard it fires at its peak
specific = ((freq > 0.0005) & (freq < 0.02)).nonzero().flatten()
chosen = specific[peak[specific].topk(min(4, len(specific))).indices].tolist()

def context(row, width=6):
    doc, pos = row // SEQ_LEN, row % SEQ_LEN
    ids = tokens[doc]
    lo, hi = max(0, pos - width), min(SEQ_LEN, pos + width + 1)
    out = []
    for i in range(lo, hi):
        s = tok.decode([ids[i].item()])
        out.append(f"[[{s}]]" if i == pos else s)
    return "".join(out).replace("\n", " ")

for feat in chosen:
    print(f"\n{'='*70}\nFEATURE {feat}  (fires on {freq[feat]*100:.2f}% of tokens)")
    top = Fv[:, feat].topk(8).indices.tolist()
    for r in top:
        print(f"  {Fv[r, feat]:5.2f}   ...{context(r)}...")
