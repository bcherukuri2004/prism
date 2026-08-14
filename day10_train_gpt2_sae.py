"""
Day 10 — Train the SAE on REAL GPT-2 activations (first run).

Same SAE as the toy, just bigger, pointed at the 32,000 real layer-6 smoothies
from Day 4. Two real-world differences from the toy:

  1. Normalize first. Real activations have an off-center mean and a big scale;
     we center them and rescale so training behaves (and so l1_coeff is sane).
  2. No answer key. Nobody knows GPT-2's "true" features, so instead of grading
     recovery we report HEALTH metrics:
       - reconstruction / variance-explained  (did it capture the activations?)
       - L0  (how many features fire per token — sparsity)
       - dead features (how many never fire at all — wasted dictionary)
"""

import torch
from sae import SparseAutoencoder, sae_loss

device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- Load and normalize the real activations ---------------------------------
d = torch.load("activations/layer6_resid.pt")
acts = d["activations"].to(device)                 # [32000, 768]
d_model = acts.shape[1]

mean = acts.mean(0)
centered = acts - mean
scale = centered.norm(dim=1).mean() / (d_model ** 0.5)   # avg L2 norm -> sqrt(d)
X = centered / scale
print(f"Normalized {X.shape[0]:,} real activations (dim {d_model}).\n")

# --- Settings ----------------------------------------------------------------
N_FEATURES = 2048     # dictionary size (~2.7x the 768 input dims)
L1_COEFF   = 2.0
LR         = 1e-3
BATCH      = 512
STEPS      = 3000

sae = SparseAutoencoder(d_model=d_model, n_features=N_FEATURES).to(device)
opt = torch.optim.Adam(sae.parameters(), lr=LR)
ever_fired = torch.zeros(N_FEATURES, dtype=torch.bool, device=device)

print("Training on real GPT-2 activations...\n")
for step in range(STEPS):
    idx = torch.randint(0, X.shape[0], (BATCH,), device=device)
    x = X[idx]
    x_hat, f = sae(x)
    total, recon, sparsity = sae_loss(x, x_hat, f, L1_COEFF)
    opt.zero_grad()
    total.backward()
    opt.step()
    with torch.no_grad():
        sae.W_dec.data /= sae.W_dec.data.norm(dim=1, keepdim=True)
        ever_fired |= (f > 0).any(dim=0)
    if step % 500 == 0 or step == STEPS - 1:
        l0 = (f > 0).float().sum(dim=1).mean().item()
        dead = (~ever_fired).sum().item()
        print(f"step {step:>4}   reconstruction={recon:8.2f}   "
              f"L0={l0:6.1f}   dead so far={dead}/{N_FEATURES}")

# --- Health metrics on the full dataset --------------------------------------
with torch.no_grad():
    recon_sum, l0_sum, n = 0.0, 0.0, 0
    fired = torch.zeros(N_FEATURES, dtype=torch.bool, device=device)
    for i in range(0, X.shape[0], 4096):
        x = X[i:i + 4096]
        x_hat, f = sae(x)
        recon_sum += (x_hat - x).pow(2).sum().item()
        l0_sum += (f > 0).float().sum().item()
        fired |= (f > 0).any(dim=0)
        n += x.shape[0]
    total_var = X.pow(2).sum().item()               # X is ~centered, so this is total variance
    fve = 1 - recon_sum / total_var                 # fraction of variance explained
    avg_l0 = l0_sum / n
    dead = (~fired).sum().item()

print("\n--- health of the real SAE ---")
print(f"variance explained: {fve*100:5.1f}%   (higher = rebuild captures more)")
print(f"active features per token (L0): {avg_l0:.1f} / {N_FEATURES}")
print(f"dead features (never fired): {dead}/{N_FEATURES}")

torch.save({"state_dict": sae.state_dict(), "mean": mean.cpu(), "scale": scale.cpu(),
            "n_features": N_FEATURES, "layer": d["layer"]}, "activations/gpt2_sae.pt")
print("\nSaved trained SAE to activations/gpt2_sae.pt")
