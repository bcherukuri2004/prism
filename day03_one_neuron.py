"""
Day 3 — See why it's confusing (superposition, up close).

Yesterday we saw the whole smoothie (768 numbers) is unreadable when blended.
Today we zoom in on ONE neuron (one gauge) and watch what makes it fire across
lots of very different sentences.

The hope: "maybe THIS gauge is the 'ocean' gauge." The reality: it lights up for
a weird, unrelated grab-bag of words — exactly like the "saltiness" gauge firing
for fries AND popcorn AND ramen. That mess is superposition, and it's the whole
reason we'll later build the SAE to un-mix things.
"""

import torch
from transformer_lens import HookedTransformer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)
print(f"Model loaded on {device}\n")

# A deliberately DIVERSE mix of topics: cooking, sports, law, ocean, code,
# animals, music, finance, space, weather, history... so a single neuron has
# the chance to fire on totally unrelated things.
sentences = [
    "The chef seared the salmon and finished it with a squeeze of lemon.",
    "She dribbled past two defenders and sank the three-pointer at the buzzer.",
    "The court ruled that the contract was void due to fraud.",
    "Waves crashed against the rocks as the tide slowly came in.",
    "He refactored the function and fixed the null pointer bug.",
    "The old oak tree was home to squirrels, owls, and a family of raccoons.",
    "The orchestra swelled as the violins carried the melody.",
    "Interest rates rose sharply after the central bank's announcement.",
    "The spacecraft entered orbit around Mars after a seven-month journey.",
    "A thunderstorm rolled in and hail battered the tin roof.",
    "In 1929 the stock market crashed and the Great Depression began.",
    "Add two cups of flour, a pinch of salt, and one egg to the bowl.",
    "The goalkeeper leapt and tipped the ball over the crossbar.",
    "The jury deliberated for six hours before reaching a verdict.",
    "Dolphins leapt alongside the boat as it sailed past the reef.",
    "The compiler threw an error on line forty-two of the script.",
    "The lion stalked the gazelle across the dry savanna.",
    "The pianist practiced the same difficult passage for hours.",
    "Quarterly profits beat expectations and the shares surged.",
    "Astronomers discovered a new exoplanet orbiting a distant star.",
    "Snow fell softly over the quiet mountain village at dawn.",
    "The Roman Empire spanned three continents at its height.",
    "Marinate the chicken overnight for the best flavor.",
    "The striker was offside, so the goal was disallowed.",
    "The lawyer objected, and the judge sustained the objection.",
    "The submarine dove beneath the icy Arctic waters.",
    "The database query timed out after thirty seconds.",
    "A pod of whales migrated south along the coastline.",
    "The choir sang the final hymn as the sun set.",
    "The investors sold their bonds and bought gold instead.",
]

LAYER = 6
HOOK = f"blocks.{LAYER}.mlp.hook_post"   # the 3,072 neuron "gauges" in layer 6

# --- Run every sentence and record each token's activations ------------------
sent_tokens = []          # str-tokens for each sentence (for showing context)
records = []              # (sentence_idx, position, token_str)
rows = []                 # activation vector [3072] for each recorded token

for s_idx, sent in enumerate(sentences):
    tokens = model.to_tokens(sent)
    str_toks = model.to_str_tokens(sent)
    sent_tokens.append(str_toks)
    _, cache = model.run_with_cache(tokens, names_filter=HOOK)
    acts = cache[HOOK][0]          # shape [seq, 3072]
    for pos in range(1, acts.shape[0]):   # skip pos 0 (the <|endoftext|> marker)
        records.append((s_idx, pos, str_toks[pos]))
        rows.append(acts[pos])

A = torch.stack(rows).float().cpu()       # [num_tokens, 3072]
print(f"Collected {A.shape[0]} tokens x {A.shape[1]} neurons.\n")

# --- Auto-pick a neuron that fires on the most DIFFERENT kinds of words -------
# (Only consider neurons that actually fire strongly, so we're not looking at noise.)
peaks = A.max(dim=0).values                          # strongest firing per neuron
threshold = torch.quantile(peaks, 0.97)              # keep only the strongest-firing neurons
candidates = (peaks >= threshold).nonzero().flatten().tolist()

def diversity(neuron):
    top = torch.topk(A[:, neuron], 8).indices.tolist()
    words = {records[i][2].strip().lower() for i in top}
    return len(words)

neuron = max(candidates, key=diversity)
print(f"Picked neuron #{neuron} in layer {LAYER} "
      f"(peak fire={peaks[neuron]:.2f}, "
      f"fires on {diversity(neuron)} different word-types in its top 8).\n")

# --- Show the words that light this one gauge up the most --------------------
def context(s_idx, pos, width=5):
    toks = sent_tokens[s_idx]
    lo, hi = max(1, pos - width), min(len(toks), pos + width + 1)
    out = []
    for i in range(lo, hi):
        out.append(f"[[{toks[i]}]]" if i == pos else toks[i])
    return "".join(out)

top = torch.topk(A[:, neuron], 12).indices.tolist()
print(f"--- Top words that fire neuron #{neuron} (the [[word]] is the trigger) ---")
for rank, i in enumerate(top, 1):
    s_idx, pos, tok = records[i]
    val = A[i, neuron].item()
    print(f"{rank:>2}. fire={val:5.2f}  trigger={tok!r:>14}   ...{context(s_idx, pos)}...")
