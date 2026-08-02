# Prism — Build Plan

**What it is:** A tool that opens up a small AI language model (GPT-2), un-mixes its
tangled internal "thoughts" into a shelf of clean, human-readable idea-features using a
*sparse autoencoder*, and lets you browse those features and *steer* the model by turning
them up and down.

**The rhythm:** Each "day" is one small session. We explain the idea → make one small
change → run it and see it work → explain what happened → next day. Smaller is always fine.

**Stack:** Python + PyTorch + TransformerLens (ML core) · FastAPI (backend) · React/Vite/Tailwind (dashboard) · Claude API (auto-labeling features).

---

## Phase 1 — Learn to see inside the model
- [x] **Day 1 — Say hello to the model.** Installed PyTorch + TransformerLens, loaded GPT-2-small on Apple Silicon (MPS), made it finish a sentence. ✅ `day01_hello_model.py`
- [x] **Day 2 — Peek at the smoothie.** Used `run_with_cache` to grab the residual stream at layer 6; saw shape (1, 10, 768) and the actual 768-number vector for the token ' ocean'. ✅ `day02_peek_inside.py`
- [x] **Day 3 — See why it's confusing.** Ran 30 diverse sentences through GPT-2, inspected layer-6 MLP neurons. Auto-picked a strongly-firing neuron and saw its top triggers were an unrelated jumble (law, cooking, code, sports, ocean) → polysemantic neuron = superposition, live. ✅ `day03_one_neuron.py`

## Phase 2 — Collect the smoothie
- [ ] **Day 4 — Pour lots of text through.** Run a text dataset through GPT-2, save one layer's activations.
- [ ] **Day 5 — Store it properly.** A simple on-disk activation store (too much for memory).

## Phase 3 — Build the un-mixer (the SAE)
- [ ] **Day 6 — Un-mix a toy first.** Tiny SAE on fake data where we know the answer.
- [ ] **Day 7 — Write the real un-mixer.** Proper SAE module (encoder + decoder) in PyTorch.
- [ ] **Day 8 — The rulebook (loss).** Reconstruction + sparsity penalty.
- [ ] **Day 9 — Teach it (small run).** Training loop on the laptop, watch loss drop.
- [ ] **Day 10 — Teach it for real (Colab).** Bigger training run, save trained weights.

## Phase 4 — Read the jars (interpretation)
- [ ] **Day 11 — What lights it up?** Max-activating examples for a feature.
- [ ] **Day 12 — Highlight the exact words.** Token-level highlighting.
- [ ] **Day 13 — Auto-label with Claude.** Name each feature via the Claude API.
- [ ] **Day 14 — Prove it's a good un-mixer.** L0, variance explained, dead features, CE recovered.

## Phase 5 — The idea knobs
- [ ] **Day 15 — Steering.** Add a feature's direction mid-generation, watch output bend.

## Phase 6 — The website
- [ ] **Day 16 — Backend.** FastAPI: fetch feature data + steered generation endpoints.
- [ ] **Day 17 — The shelf.** React app + feature-browser page.
- [ ] **Day 18 — Examples view.** Render highlighted examples.
- [ ] **Day 19 — The scoreboard.** Charts for eval metrics.
- [ ] **Day 20 — The playground.** Live steering knobs in the browser.
- [ ] **Day 21 — Polish & write-up.** README, explainer, screenshots.

---

### Progress log
- **Day 1 (done):** Created project folder + fresh Python 3.13 venv on Apple Silicon (MPS). Installed torch 2.13 + transformer_lens 3.6. Wrote `day01_hello_model.py`: loads GPT-2-small on MPS and completes a sentence. Verified working. Noticed GPT-2 repeats itself — normal for a small 2019 model.
- **Day 2 (done):** Wrote `day02_peek_inside.py`. Learned: tokens (sentence → word-chunks), the residual stream (the model's "conveyor belt" of thinking). Used `run_with_cache` to grab layer-6 residual: shape (1, 10, 768) = 1 sentence × 10 tokens × 768 numbers each. Printed the real activation vector for ' ocean'. This is "the smoothie" — real but still blended (SAE later un-mixes it).
- **Day 2.5 (concept):** Nailed vocabulary: tensor = a box of numbers (the smoothie is one tensor). layer = a worker/station (12 in GPT-2-small, fixed by blueprint). neuron = one gauge/number-slot inside a layer (3,072 per layer). model = the "company." 768 = width of one token's smoothie (varies by model), NOT the neuron count. Saved `docs/concept-map.svg`.
- **Day 3 (done):** Wrote `day03_one_neuron.py`. Ran 30 topic-diverse sentences, hooked `blocks.6.mlp.hook_post` (3,072 neurons), auto-picked a strong-firing neuron. Its top triggers were an unrelated grab-bag (objected/Add/due/threw/beneath/leapt...) spanning law, cooking, code, sports, ocean. Concrete proof of a polysemantic neuron → superposition. This is *why* single neurons are unreadable and why the SAE is needed.
