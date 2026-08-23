"""
Day 13 — Auto-label features with Claude.

For each feature we gather the text snippets that make it fire hardest, hand them
to Claude, and ask for a short human name ("patent boilerplate", "closing HTML
tag"). This is "auto-interp": using a big model to interpret a small one's
features, so we don't hand-label thousands of them.

The API key is read from prism/.env (ANTHROPIC_API_KEY) — never hard-coded.
"""

import json
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer
from sae import SparseAutoencoder
import anthropic

load_dotenv()                      # loads prism/.env -> ANTHROPIC_API_KEY
MODEL = "claude-opus-5"            # default per the API guide; a cheaper model (claude-haiku-4-5)
                                   # would be sensible when labeling ALL ~2000 features.

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Load trained SAE + data (same as Day 11/12) -----------------------------
ckpt = torch.load("activations/gpt2_sae.pt")
sae = SparseAutoencoder(d_model=768, n_features=ckpt["n_features"]).to(device)
sae.load_state_dict(ckpt["state_dict"])
mean, scale = ckpt["mean"].to(device), ckpt["scale"].to(device)

d = torch.load("activations/layer6_resid.pt")
acts, tokens = d["activations"], d["tokens"]
SEQ_LEN = tokens.shape[1]
tok = AutoTokenizer.from_pretrained("gpt2")

print("Computing feature activations...")
feats = []
with torch.no_grad():
    for i in range(0, acts.shape[0], 4096):
        x = (acts[i:i + 4096].to(device) - mean) / scale
        feats.append(sae.encode(x).cpu())
F = torch.cat(feats)
valid = torch.tensor([r % SEQ_LEN != 0 for r in range(F.shape[0])])
Fv = F.clone(); Fv[~valid] = 0.0

freq = (Fv > 0).float().mean(dim=0)
peak = Fv.max(dim=0).values
specific = ((freq > 0.0005) & (freq < 0.02)).nonzero().flatten()
chosen = specific[peak[specific].topk(min(6, len(specific))).indices].tolist()

def snippet(row, width=7):
    doc, pos = row // SEQ_LEN, row % SEQ_LEN
    ids = tokens[doc]
    lo, hi = max(0, pos - width), min(SEQ_LEN, pos + width + 1)
    out = []
    for i in range(lo, hi):
        s = tok.decode([ids[i].item()])
        out.append(f"«{s}»" if i == pos else s)
    return "".join(out).replace("\n", " ")

# --- Ask Claude to name each feature -----------------------------------------
client = anthropic.Anthropic()
SYSTEM = (
    "You are an interpretability researcher labeling features from a sparse "
    "autoencoder trained on GPT-2. You are given text snippets that make one "
    "feature fire; the exact triggering token is wrapped in «». Respond with "
    "ONLY a short label (2-6 words) naming the concept, then ' — ', then a "
    "one-sentence description. No preamble. Do not include any XML or system tags."
)

labels = {}
print(f"\nLabeling {len(chosen)} features with {MODEL}...\n")
for feat in chosen:
    rows = Fv[:, feat].topk(10).indices.tolist()
    user = "Snippets (triggering token in «»):\n" + "\n".join(f"- {snippet(r)}" for r in rows)
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=200,
            thinking={"type": "disabled"},          # simple labeling — no need to think
            output_config={"effort": "low"},
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError:
        print("Auth failed — check ANTHROPIC_API_KEY in prism/.env"); raise
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    labels[str(feat)] = text
    print(f"  Feature {feat:>4}:  {text}")

with open("docs/feature_labels.json", "w") as f:
    json.dump(labels, f, indent=2)
print("\nSaved labels to docs/feature_labels.json")
