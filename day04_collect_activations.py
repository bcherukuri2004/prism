"""
Day 4 — Pour lots of text through and save the smoothies.

The SAE (our un-mixer, built later) learns by looking at MANY real smoothies.
So today we run a real text dataset through GPT-2, scoop the residual-stream
smoothie for every token at one layer, and save the whole pile to disk.

No new ideas — just Day 2's "grab the smoothie" move, done thousands of times.
"""

import os
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)   # we're only running the model, not training it — saves memory

# --- Settings (small on purpose; we scale up in Colab later) -----------------
LAYER    = 6      # which layer's smoothie to collect (the belt after worker #6)
SEQ_LEN  = 128    # how many tokens we keep from each document
N_SEQ    = 250    # how many documents -> 250 * 128 = 32,000 token-smoothies
BATCH    = 16     # documents processed at once
HOOK     = f"blocks.{LAYER}.hook_resid_post"   # the residual stream after layer 6
OUT_PATH = "activations/layer6_resid.pt"

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)
print(f"Model loaded on {device}\n")

# --- Step 1: load a real text dataset ----------------------------------------
# "pile-10k" = 10,000 documents from The Pile, a standard research corpus.
# First run downloads it (~tens of MB) and caches it.
print("Loading dataset (first run downloads it)...")
dataset = load_dataset("NeelNanda/pile-10k", split="train")
print(f"Dataset has {len(dataset)} documents.\n")

# --- Step 2: turn documents into uniform 128-token chunks --------------------
# We keep only documents long enough to give us a full 128 tokens, and cut them
# to exactly 128 so every chunk is the same shape (easy to batch, no padding).
print(f"Tokenizing until we have {N_SEQ} chunks of {SEQ_LEN} tokens...")
chunks = []
for row in dataset:
    toks = model.to_tokens(row["text"])          # [1, seq]  (includes a start marker)
    if toks.shape[1] >= SEQ_LEN:
        chunks.append(toks[0, :SEQ_LEN])
    if len(chunks) >= N_SEQ:
        break
token_batch = torch.stack(chunks)                 # [N_SEQ, SEQ_LEN]
print(f"Got {token_batch.shape[0]} chunks, each {token_batch.shape[1]} tokens.\n")

# --- Step 3: run the model and scoop the smoothie at LAYER, in batches --------
print("Running GPT-2 and collecting smoothies...")
all_smoothies = []
for i in range(0, token_batch.shape[0], BATCH):
    batch = token_batch[i:i + BATCH].to(device)
    _, cache = model.run_with_cache(batch, names_filter=HOOK)
    smoothies = cache[HOOK]                        # [batch, SEQ_LEN, 768]
    smoothies = smoothies.reshape(-1, smoothies.shape[-1])   # [batch*SEQ_LEN, 768]
    all_smoothies.append(smoothies.cpu())
    print(f"  processed {min(i + BATCH, token_batch.shape[0])}/{token_batch.shape[0]} chunks")

activations = torch.cat(all_smoothies, dim=0)      # [total_tokens, 768]

# --- Step 4: save the pile to disk -------------------------------------------
os.makedirs("activations", exist_ok=True)
torch.save(
    {"activations": activations, "tokens": token_batch, "layer": LAYER},
    OUT_PATH,
)
size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"\nDone. Saved {activations.shape[0]:,} smoothies "
      f"(each {activations.shape[1]} numbers) to {OUT_PATH} ({size_mb:.0f} MB).")
