"""
Day 7 — Build the un-mixer (the sparse autoencoder), untrained.

An SAE has two halves:
  - encoder: smoothie  -> feature activations   ("which features are on?")
  - decoder: features  -> rebuilt smoothie      ("put the mix back together")

If the encoder un-mixed correctly, the decoder's rebuild should match the
original smoothie. Training (Day 9) nudges both halves until that's true.
Today we just build the machine and confirm it runs — untrained, so its output
is nonsense on purpose. This file is reusable; later days import it.
"""

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """A basic sparse autoencoder.

    d_model    = size of each smoothie (toy: 20, real GPT-2: 768)
    n_features = size of the feature dictionary (how many features we can find).
                 Made LARGER than the true concepts so each feature can be clean.
    """

    def __init__(self, d_model, n_features):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        # Decoder directions: one row per feature = that feature's "shape" in
        # smoothie-space. Start random, then normalize each to unit length.
        self.W_dec = nn.Parameter(torch.randn(n_features, d_model))
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

        # Encoder starts as the decoder's transpose (a common, sensible starting point).
        self.W_enc = nn.Parameter(self.W_dec.data.t().clone())   # [d_model, n_features]
        self.b_enc = nn.Parameter(torch.zeros(n_features))       # encoder bias
        self.b_dec = nn.Parameter(torch.zeros(d_model))          # subtracted before / added after

    def encode(self, x):
        """Smoothie -> feature activations. ReLU keeps activations >= 0 (a feature
        is either off or on-by-some-amount, never negative)."""
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f):
        """Feature activations -> rebuilt smoothie (a weighted sum of feature shapes)."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)          # which features are on
        x_hat = self.decode(f)      # rebuild the smoothie from them
        return x_hat, f


def sae_loss(x, x_hat, f, l1_coeff):
    """The grading rule (Day 8). Two parts that pull against each other:

      reconstruction = how close the rebuild x_hat is to the original x  (want LOW)
      sparsity (L1)  = total feature activity per smoothie                (want LOW)

    total = reconstruction + l1_coeff * sparsity
    l1_coeff sets how hard we push for fewer features. Bigger = sparser but blurrier
    rebuilds; smaller = sharper rebuilds but messier (denser) features.
    """
    reconstruction = (x_hat - x).pow(2).sum(dim=1).mean()   # MSE per smoothie, averaged
    sparsity       = f.abs().sum(dim=1).mean()              # L1: total feature use, averaged
    total          = reconstruction + l1_coeff * sparsity
    return total, reconstruction, sparsity


if __name__ == "__main__":
    # Load the toy world from Day 6 and run the untrained SAE on it.
    toy = torch.load("toy/toy_data.pt")
    data = toy["data"]                       # [5000, 20]
    d_model = data.shape[1]

    sae = SparseAutoencoder(d_model=d_model, n_features=16)
    print(f"Built an SAE: {d_model}-D smoothies -> {sae.n_features} features -> {d_model}-D rebuild")
    total_params = sum(p.numel() for p in sae.parameters())
    print(f"It has {total_params} adjustable knobs (weights).\n")

    x_hat, f = sae(data)                     # run the whole dataset through

    print(f"feature activations shape: {tuple(f.shape)}  (one row per smoothie, 16 features each)")
    print(f"rebuilt smoothies shape:   {tuple(x_hat.shape)}\n")

    # How wrong is the rebuild right now? Compare to a do-nothing baseline
    # (always predict the average smoothie), which is what 'no skill' looks like.
    untrained_error = (x_hat - data).pow(2).mean().item()
    baseline_error  = (data.mean(0) - data).pow(2).mean().item()
    print(f"Rebuild error, UNTRAINED SAE: {untrained_error:.4f}")
    print(f"Rebuild error, do-nothing baseline: {baseline_error:.4f}")
    print("\n^ Untrained, the SAE is no better than guessing the average — as expected.")
    print("  Day 8 adds the 'grading rule' (loss); Day 9 trains it until the rebuild matches.")
