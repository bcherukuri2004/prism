"""
Day 2 — Peek at the smoothie.

Goal: while GPT-2 reads a sentence, reach inside and grab the numbers on its
"residual stream" (the conveyor belt where its thinking lives), and look at them.

Yesterday: text in -> text out (sealed box).
Today:     text in -> we SEE the numbers in the middle.
"""

import torch
from transformer_lens import HookedTransformer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)
print(f"Model loaded on {device}\n")

sentence = "I love walking along the ocean at sunset."

# --- Step 1: see how the sentence gets chopped into "tokens" -------------------
# A model doesn't read words, it reads TOKENS (word-chunks). Let's see them.
str_tokens = model.to_str_tokens(sentence)
print(f"Sentence: {sentence!r}")
print(f"Chopped into {len(str_tokens)} tokens: {str_tokens}\n")

# --- Step 2: run the model, but ask it to SAVE everything it computes ----------
# run_with_cache = "run the model AND keep a recording of every internal number."
# The 'cache' is that recording. This is the x-ray glasses in action.
tokens = model.to_tokens(sentence)
logits, cache = model.run_with_cache(tokens)

# --- Step 3: grab the residual stream at a middle layer (layer 6 of 0..11) -----
# This is THE SMOOTHIE: what the model is "thinking" at that point.
layer = 6
resid = cache["resid_post", layer]

print(f"--- The smoothie at layer {layer} ---")
print(f"Shape of what we grabbed: {tuple(resid.shape)}")
print("Reading that shape: [1 sentence,  {} tokens,  {} numbers per token]".format(
    resid.shape[1], resid.shape[2]))

# --- Step 4: actually LOOK at a piece of it -----------------------------------
# Pull the smoothie for one token (the word ' ocean') and show the first 8 numbers.
ocean_index = str_tokens.index(" ocean")
ocean_vector = resid[0, ocean_index]
print(f"\nThe token ' ocean' is token #{ocean_index}.")
print(f"Its smoothie is a list of {ocean_vector.shape[0]} numbers. First 8 of them:")
print(ocean_vector[:8].tolist())
