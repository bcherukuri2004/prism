"""
Day 14 — Prove it's a good un-mixer (the standard SAE health metrics).

Four numbers researchers use to judge an SAE:
  - variance explained : does the rebuild capture the activations? (from Day 10)
  - L0                 : how many features fire per token (sparsity)
  - dead features      : how many features never fire (wasted dictionary)
  - CE loss recovered  : the gold standard. Swap the model's real layer-6
                         smoothies for the SAE's REBUILT ones and see if GPT-2
                         can still predict the next word. ~100% = the rebuild
                         preserves what the model actually uses.
"""

import json
import torch
from transformer_lens import HookedTransformer
from sae import SparseAutoencoder

torch.set_grad_enabled(False)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Load model, SAE, data ---------------------------------------------------
model = HookedTransformer.from_pretrained("gpt2", device=device)
ckpt = torch.load("activations/gpt2_sae.pt")
sae = SparseAutoencoder(768, ckpt["n_features"]).to(device)
sae.load_state_dict(ckpt["state_dict"])
mean, scale = ckpt["mean"].to(device), ckpt["scale"].to(device)
LAYER = ckpt["layer"]
HOOK = f"blocks.{LAYER}.hook_resid_post"

d = torch.load("activations/layer6_resid.pt")
acts, tokens = d["activations"], d["tokens"]
BATCH = 8

# --- Variance explained, L0, dead features (recomputed) ----------------------
recon_sum, l0_sum, n = 0.0, 0.0, 0
fired = torch.zeros(ckpt["n_features"], dtype=torch.bool, device=device)
X = (acts.to(device) - mean) / scale
for i in range(0, X.shape[0], 4096):
    x = X[i:i + 4096]
    x_hat, f = sae(x)
    recon_sum += (x_hat - x).pow(2).sum().item()
    l0_sum += (f > 0).float().sum().item()
    fired |= (f > 0).any(dim=0)
    n += x.shape[0]
fve = 1 - recon_sum / X.pow(2).sum().item()
avg_l0 = l0_sum / n
dead = (~fired).sum().item()

# --- CE loss recovered -------------------------------------------------------
def sae_hook(value, hook):                       # replace resid with SAE rebuild
    x = (value - mean) / scale
    x_hat = sae(x)[0]
    return x_hat * scale + mean

def mean_hook(value, hook):                      # baseline: destroy resid -> dataset mean
    return mean.expand_as(value)

def avg_loss(fwd_hooks=None):
    total, seqs = 0.0, 0
    for i in range(0, tokens.shape[0], BATCH):
        batch = tokens[i:i + BATCH].to(device)
        if fwd_hooks is None:
            loss = model(batch, return_type="loss")
        else:
            loss = model.run_with_hooks(batch, return_type="loss", fwd_hooks=fwd_hooks)
        total += loss.item() * batch.shape[0]
        seqs += batch.shape[0]
    return total / seqs

print("Measuring cross-entropy loss (clean / SAE-patched / mean-ablated)...")
L_clean = avg_loss()
L_sae   = avg_loss([(HOOK, sae_hook)])
L_mean  = avg_loss([(HOOK, mean_hook)])
ce_recovered = (L_mean - L_sae) / (L_mean - L_clean)

# --- Report ------------------------------------------------------------------
metrics = {
    "layer": LAYER,
    "dictionary_size": ckpt["n_features"],
    "variance_explained_pct": round(fve * 100, 1),
    "L0_active_features": round(avg_l0, 1),
    "dead_features": int(dead),
    "ce_loss_clean": round(L_clean, 3),
    "ce_loss_sae_patched": round(L_sae, 3),
    "ce_loss_mean_ablated": round(L_mean, 3),
    "ce_loss_recovered_pct": round(ce_recovered * 100, 1),
}
print("\n--- Prism SAE metrics (layer 6) ---")
print(f"dictionary size            : {metrics['dictionary_size']}")
print(f"variance explained         : {metrics['variance_explained_pct']}%")
print(f"active features per token   : {metrics['L0_active_features']}")
print(f"dead features               : {metrics['dead_features']}/{metrics['dictionary_size']}")
print(f"CE loss (clean)             : {metrics['ce_loss_clean']}")
print(f"CE loss (SAE-patched)       : {metrics['ce_loss_sae_patched']}")
print(f"CE loss (mean-ablated)      : {metrics['ce_loss_mean_ablated']}")
print(f"CE loss recovered           : {metrics['ce_loss_recovered_pct']}%  (100% = rebuild is as good as the real thing)")

with open("docs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nSaved docs/metrics.json")
