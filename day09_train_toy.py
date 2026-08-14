"""
Day 9 — Train the un-mixer on the toy world, and grade it.

We finally teach the SAE. The training loop repeats thousands of times:
  1. grab a batch of toy smoothies
  2. SAE un-mixes -> features -> rebuilds
  3. grade it (reconstruction + sparsity)
  4. nudge the knobs to lower the grade
Then we CHECK ITS WORK: did the SAE's learned features line up with the 8
secret features we planted on Day 6? (We can only do this because it's a toy.)
"""

import torch
from sae import SparseAutoencoder, sae_loss

device = "mps" if torch.backends.mps.is_available() else "cpu"

toy = torch.load("toy/toy_data.pt")
data          = toy["data"].to(device)          # [5000, 20]
true_features = toy["true_features"]             # [8, 20]  (the answer key)

# --- Settings ----------------------------------------------------------------
N_FEATURES = 16       # dictionary size (bigger than the 8 real ones on purpose)
L1_COEFF   = 0.05     # how hard we push for sparsity
LR         = 1e-3
BATCH      = 256
STEPS      = 4000

sae = SparseAutoencoder(d_model=data.shape[1], n_features=N_FEATURES).to(device)
opt = torch.optim.Adam(sae.parameters(), lr=LR)

print("Training the toy SAE...\n")
for step in range(STEPS):
    idx = torch.randint(0, data.shape[0], (BATCH,), device=device)
    x = data[idx]

    x_hat, f = sae(x)
    total, recon, sparsity = sae_loss(x, x_hat, f, L1_COEFF)

    opt.zero_grad()
    total.backward()
    opt.step()

    # Keep decoder rows unit-length so the L1 penalty can't be "cheated"
    # by scaling features down and decoder up.
    with torch.no_grad():
        sae.W_dec.data /= sae.W_dec.data.norm(dim=1, keepdim=True)

    if step % 800 == 0 or step == STEPS - 1:
        l0 = (f > 0).float().sum(dim=1).mean().item()
        print(f"step {step:>4}   reconstruction={recon:6.3f}   "
              f"sparsity(L1)={sparsity:5.3f}   active features (L0)={l0:4.1f}")

# --- Grade it: did it recover our 8 planted features? ------------------------
with torch.no_grad():
    x_hat, f = sae(data)
    final_recon = (x_hat - data).pow(2).sum(dim=1).mean().item()
    final_l0 = (f > 0).float().sum(dim=1).mean().item()

    learned = sae.W_dec.detach().cpu()                    # [16, 20], unit rows
    truth   = true_features                               # [8, 20],  unit rows
    cos = truth @ learned.t()                             # [8, 16] cosine similarity
    best_match = cos.max(dim=1).values                    # best learned match per true feature

print("\n--- did it work? ---")
print(f"final reconstruction error: {final_recon:.4f}  (was ~0.49 untrained)")
print(f"final active features (L0): {final_l0:.2f}  (answer key: ~2)")
print("\nHow well each of the 8 planted features was recovered (1.00 = perfect):")
for i, c in enumerate(best_match.tolist()):
    bar = "#" * int(c * 20)
    print(f"  true feature {i}:  {c:.3f}  {bar}")
recovered = (best_match > 0.9).sum().item()
print(f"\n{recovered}/8 planted features recovered with >0.90 similarity.")

torch.save(sae.state_dict(), "toy/toy_sae.pt")
