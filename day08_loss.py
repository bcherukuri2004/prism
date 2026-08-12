"""
Day 8 — The grading rule (loss), demonstrated.

We import the SAE and its loss from sae.py, run the UNTRAINED SAE on the toy
data, and look at the two parts of the grade plus the tension between them.

  total loss = reconstruction  +  l1_coeff * sparsity
               (rebuild badness)         (how many features are on)

Nothing is trained yet — this just shows what the trainer (Day 9) will push down.
"""

import torch
from sae import SparseAutoencoder, sae_loss

toy = torch.load("toy/toy_data.pt")
data = toy["data"]                       # [5000, 20]

sae = SparseAutoencoder(d_model=data.shape[1], n_features=16)
x_hat, f = sae(data)

# L0 = the ACTUAL number of active features per smoothie (what sparsity targets).
# Our answer key says ~2 should be active; an untrained SAE won't be near that.
l0 = (f > 0).float().sum(dim=1).mean().item()

print("--- grading the UNTRAINED SAE on the toy data ---")
for l1_coeff in [0.0, 0.01, 0.1]:
    total, recon, sparsity = sae_loss(data, x_hat, f, l1_coeff)
    print(f"l1_coeff={l1_coeff:<4}  total={total:6.3f}   "
          f"reconstruction={recon:6.3f}   sparsity(L1)={sparsity:6.3f}")

print(f"\nActive features per smoothie right now (L0): {l0:.1f} out of 16")
print("The answer key says only ~2 features should really be on per smoothie.")
print("\nThe tension, in one line:")
print("  * push l1_coeff too HIGH -> features all switch off -> rebuild gets worse")
print("  * push l1_coeff too LOW  -> uses tons of features   -> features stay messy")
print("  Day 9 trains the knobs to make BOTH numbers small at once.")
