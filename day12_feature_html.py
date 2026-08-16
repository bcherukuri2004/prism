"""
Day 12 — Token-level highlighting: render features as a readable dashboard.

Day 11 marked only the single peak token. Here we shade EVERY token in each
example by how strongly the feature fires on it (a mini heatmap), and write it
all to a self-contained HTML page. This is the first piece of the Phase-6 UI.
"""

import html as htmllib
import torch
from transformers import AutoTokenizer
from sae import SparseAutoencoder

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Load trained SAE + data (same as Day 11) --------------------------------
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
F = torch.cat(feats)                                  # [32000, 2048]

valid = torch.tensor([r % SEQ_LEN != 0 for r in range(F.shape[0])])
Fv = F.clone(); Fv[~valid] = 0.0

freq = (Fv > 0).float().mean(dim=0)
peak = Fv.max(dim=0).values
specific = ((freq > 0.0005) & (freq < 0.02)).nonzero().flatten()
chosen = specific[peak[specific].topk(min(6, len(specific))).indices].tolist()

# --- Build the HTML ----------------------------------------------------------
def token_span(row, feat, maxact):
    a = max(0.0, F[row, feat].item()) / maxact
    s = htmllib.escape(tok.decode([tokens[row // SEQ_LEN, row % SEQ_LEN].item()]))
    s = s.replace("\n", "⏎")
    weight = "font-weight:600;" if a > 0.5 else ""
    return f'<span style="background:rgba(255,145,0,{a:.2f});{weight}">{s}</span>'

cards = []
for feat in chosen:
    maxact = peak[feat].item()
    rows = Fv[:, feat].topk(10).indices.tolist()
    examples = []
    for r in rows:
        doc, pos = r // SEQ_LEN, r % SEQ_LEN
        lo, hi = max(0, pos - 7), min(SEQ_LEN, pos + 8)
        window = "".join(token_span(doc * SEQ_LEN + i, feat, maxact) for i in range(lo, hi))
        examples.append(f'<div class="ex">{window}</div>')
    cards.append(f"""
      <div class="card">
        <div class="head">Feature {feat}
          <span class="meta">fires on {freq[feat]*100:.2f}% of tokens · label: <em>?</em></span>
        </div>
        {''.join(examples)}
      </div>""")

page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Prism — feature dashboard</title>
<style>
  body {{ font: 15px/1.5 -apple-system, Helvetica, Arial, sans-serif; background:#faf9f7; color:#222; margin:0; padding:24px; }}
  h1 {{ font-size:20px; font-weight:600; }}
  .sub {{ color:#666; margin-bottom:20px; }}
  .card {{ background:#fff; border:1px solid #e5e3df; border-radius:10px; padding:14px 16px; margin-bottom:14px; }}
  .head {{ font-weight:600; margin-bottom:8px; }}
  .meta {{ font-weight:400; color:#888; font-size:13px; margin-left:8px; }}
  .ex {{ white-space:pre-wrap; font-family:ui-monospace, Menlo, monospace; font-size:13px;
         padding:3px 0; border-top:1px solid #f0eee9; }}
</style></head><body>
  <h1>Prism — feature dashboard</h1>
  <div class="sub">Each token is shaded by how strongly the feature fires on it. Layer 6 · trained SAE.</div>
  {''.join(cards)}
</body></html>"""

with open("docs/feature_dashboard.html", "w") as f:
    f.write(page)
print(f"Wrote docs/feature_dashboard.html with {len(chosen)} features: {chosen}")
