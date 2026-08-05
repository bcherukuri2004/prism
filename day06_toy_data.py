"""
Day 6 — Build a toy world where we KNOW the answer.

The real SAE has a scary problem: when it un-mixes a GPT-2 smoothie, nobody knows
the "true" features, so we can't grade it. Fix: first build a fake world where WE
plant the ingredients, so we have a perfect answer key.

We invent:
  - true_features: a small set of secret "ingredient" directions (the clean features)
  - codes:         for each fake smoothie, WHICH features are active and how strongly
  - data:          the fake smoothies = a sparse mix of those features (what the SAE sees)

Later the SAE will get only `data` and must recover `true_features` and `codes`.
Because we saved the answer key, we'll be able to check it exactly.
"""

import os
import torch

torch.manual_seed(0)   # reproducible: same toy world every run

# --- Settings (tiny on purpose, so we can inspect everything) -----------------
D_MODEL     = 20     # size of each fake "smoothie" (toy version of GPT-2's 768)
N_FEATURES  = 8      # how many secret ingredients exist in this world
N_SAMPLES   = 5000   # how many fake smoothies to make
AVG_ACTIVE  = 2      # sparsity: on average ~2 of the 8 features are active per smoothie
NOISE       = 0.01   # a tiny bit of measurement noise

# --- The secret ingredients: 8 random directions in 20-D space ----------------
# Each is normalized to length 1 so they're comparable. This is the ANSWER KEY.
true_features = torch.randn(N_FEATURES, D_MODEL)
true_features = true_features / true_features.norm(dim=1, keepdim=True)   # [8, 20]

# --- Make the fake smoothies: sparse mixes of the ingredients -----------------
# codes[i, f] = how strongly feature f is active in smoothie i (0 = off).
# Each feature turns on with probability AVG_ACTIVE/N_FEATURES, at a random strength.
active   = (torch.rand(N_SAMPLES, N_FEATURES) < AVG_ACTIVE / N_FEATURES).float()
strength = torch.rand(N_SAMPLES, N_FEATURES)          # random positive magnitudes
codes    = active * strength                           # [5000, 8]  (mostly zeros = sparse)

# data = mix the active ingredients together, plus a little noise.
data = codes @ true_features                           # [5000, 20]
data = data + NOISE * torch.randn_like(data)

# --- Save the whole toy world (data + answer key) -----------------------------
os.makedirs("toy", exist_ok=True)
torch.save({"data": data, "true_features": true_features, "codes": codes}, "toy/toy_data.pt")

# --- Show what we made --------------------------------------------------------
avg_active = (codes > 0).float().sum(dim=1).mean().item()
print(f"Made {N_SAMPLES} fake smoothies, each {D_MODEL} numbers wide.")
print(f"Secret ingredients (true features): {N_FEATURES}")
print(f"Average active features per smoothie: {avg_active:.2f}  (we aimed for ~{AVG_ACTIVE})\n")

print("--- Look at smoothie #0 ---")
print("Its ANSWER KEY (which of the 8 features are on, and how strong):")
for f in range(N_FEATURES):
    on = codes[0, f].item()
    print(f"   feature {f}: {'ON  ' if on > 0 else 'off '} {on:.2f}")
print("\nBut the SAE only gets to see the mixed smoothie (20 numbers):")
print("  ", [round(x, 2) for x in data[0].tolist()])
print("\n^ You can't read the answer key off those 20 numbers — that's the whole point.")
print("  The SAE's job (built next) is to recover the answer key from the mix.")
