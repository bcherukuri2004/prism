"""
Day 1 — Say hello to the model.

Goal: load GPT-2-small and make it finish a sentence.

Big picture: GPT-2 is a "language model" — a program that, given some text,
predicts what word (really, what *token*) most likely comes next. That's the
entire trick behind ChatGPT and friends: predict the next token, over and over.

We use TransformerLens (a library made specifically for LOOKING INSIDE models)
instead of the plain HuggingFace library, because in a few days we'll want to
reach into the model's guts. Might as well start with the right tool.
"""

import torch
from transformer_lens import HookedTransformer

# 1) Pick where the math runs.
#    - "mps" = Apple Silicon GPU (your Mac has this). Faster.
#    - "cpu" = fallback that always works.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2) Load GPT-2-small. The FIRST time, this downloads ~500MB of weights from
#    HuggingFace and caches them, so future runs are instant.
#    "HookedTransformer" is just GPT-2 with easy hooks for peeking inside later.
print("Loading GPT-2-small (first run downloads weights, please wait)...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
print("Model loaded!\n")

# 3) Give it the start of a sentence and let it continue.
prompt = "The best thing about living near the ocean is"
print(f"Prompt: {prompt!r}")

# generate() runs "predict the next token" repeatedly, 30 times here.
# temperature controls randomness: 0.7 = a little creative but coherent.
output = model.generate(
    prompt,
    max_new_tokens=30,
    temperature=0.7,
    verbose=False,
)

print("\n--- GPT-2 completed it with: ---")
print(output)
