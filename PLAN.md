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
- [x] **Day 4 — Pour lots of text through.** Ran 250 docs (128 tokens each) from The Pile through GPT-2, scooped the layer-6 residual smoothie for all 32,000 tokens, saved to `activations/layer6_resid.pt` (94 MB, shape 32000×768). ✅ `day04_collect_activations.py`
- [x] **Day 5 — Store it properly.** Built `activation_store.py`: `build_shards()` splits the pile into 4 disk shards; `ActivationStore` streams shuffled batches via a memory-capped buffer. Demo served 500 batches of 64 without holding >16k rows in RAM. ✅ (reusable — Day 9 training imports it)

## Phase 3 — Build the un-mixer (the SAE)
- [x] **Day 6 — Build the toy world.** Planted 8 secret "true features" (unit directions in 20-D), generated 5,000 sparse mixes (~2 active each) + saved the answer key (`codes`). This gives a gradable test bed for the SAE. ✅ `day06_toy_data.py` → `toy/toy_data.pt`
  - *(Days 7–9 build & train the SAE, validating on this toy first, then real data.)*
- [x] **Day 7 — Write the un-mixer.** Built `SparseAutoencoder` (encoder: ReLU((x−b_dec)·W_enc+b_enc); decoder: f·W_dec+b_dec; unit-norm decoder rows, encoder init = decoder transpose). Ran untrained on toy data (20→16→20): reconstruction ≈ do-nothing baseline, as expected. ✅ `sae.py` (reusable)
- [x] **Day 8 — The rulebook (loss).** Added `sae_loss` to `sae.py`: total = reconstruction (MSE/smoothie) + l1_coeff·sparsity (L1 of feature acts). Demo (`day08_loss.py`) on untrained SAE: recon 0.490, L1 0.849, L0 7.5/16 active (answer key wants ~2). Shows the accuracy-vs-sparsity tension the trainer must balance. ✅
- [x] **Day 9 — Teach it (small run).** Wrote `day09_train_toy.py`: Adam training loop (l1_coeff 0.05, lr 1e-3, 4000 steps) with per-step decoder unit-norm. Reconstruction 0.49→0.0048, L0 6.9→~3. **Recovered all 8/8 planted features at >0.99 cosine similarity.** The un-mixer works, proven on data with a known answer. ✅ → `toy/toy_sae.pt`
- [x] **Day 10 — Teach it for real.** Trained the SAE on the 32k real GPT-2 layer-6 activations (locally, MPS). Normalized inputs (center + scale to avg norm √d). 768→2048→768, l1=2.0, 3000 steps. Result: **98.2% variance explained, L0≈36, only 10/2048 dead.** Saved `activations/gpt2_sae.pt`. ✅ `day10_train_gpt2_sae.py`  *(Colab = optional bigger/higher-quality run: more tokens, 8–32× dict.)*

## Phase 4 — Read the jars (interpretation)
- [x] **Day 11 — What lights it up?** Wrote `day11_feature_examples.py`: ran SAE encoder over all 32k tokens, found max-activating examples per feature. Discovered **crisply interpretable features**: #1225 = patent "The present invention relates to…"; #436 = closing HTML/XML tags `</`; #65 = opening curly-quote; #433 = "Q:" forum question header. The SAE genuinely un-mixed real GPT-2 concepts. ✅
- [x] **Day 12 — Highlight the exact words.** Wrote `day12_feature_html.py`: shades every token by its activation (mini heatmap) and renders a self-contained `docs/feature_dashboard.html` (6 features). First piece of the Phase-6 UI. ✅
- [x] **Day 13 — Label features.** Wrote `day13_autolabel.py` (auto-interp via Claude API, ready + reads gitignored `.env`). No live API key available, so labeled the 6 features by hand in `docs/feature_labels.json`: #1225 patent "invention", #1901 patent "present", #436 closing XML/HTML tag, #433 forum "Q:" header, #65 opening curly-quote, #1164 closing curly-quote. ✅
- [x] **Day 14 — Prove it's a good un-mixer.** Wrote `day14_metrics.py`. Full metrics: **98.2% variance explained, L0 35.8, 10/2048 dead, and 91.2% cross-entropy loss recovered** (clean 3.57 → SAE-patched 4.13 vs mean-ablated 9.91). Saved `docs/metrics.json`. Phase 4 complete. ✅

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
- **Day 4 (done):** Wrote `day04_collect_activations.py`. Loaded `NeelNanda/pile-10k`, kept 250 docs cut to 128 tokens, batched through GPT-2, collected `blocks.6.hook_resid_post` for all 32,000 tokens → tensor 32000×768, saved to `activations/layer6_resid.pt` (94 MB, gitignored). This pile is the SAE's training data. Note: residual stream (768 dims), not MLP neurons — the SAE targets the belt.
- **Day 5 (done):** Wrote `activation_store.py` (reusable module). `build_shards()` cut the pile into 4×23 MB shards; `ActivationStore.batches()` streams shuffled [64,768] batches using a 16k-row shuffle buffer, loading shards lazily. Demo: 500 batches served, memory capped. Phase 2 complete — data is ready for the SAE.
- **Day 6 (done):** Wrote `day06_toy_data.py`. Built a synthetic test bed with a known answer key: 8 true features (unit vecs in 20-D), 5,000 samples = sparse mixes (~1.97 active avg) + tiny noise. Saved `data`, `true_features`, `codes` to `toy/toy_data.pt`. Purpose: the real SAE can't be graded (no ground truth for GPT-2), so we validate the SAE on this toy first. `torch.manual_seed(0)` for reproducibility.
- **Day 7 (done):** `sae.py` — `SparseAutoencoder` (encoder ReLU((x−b_dec)·W_enc+b_enc), decoder f·W_dec+b_dec, unit-norm decoder rows, encoder init = decoder.T). Untrained ≈ baseline.
- **Day 8 (done):** Added `sae_loss` (recon MSE + l1_coeff·L1). `day08_loss.py` showed untrained L0 7.5/16 vs toy's ~2 → the accuracy-vs-sparsity tension.
- **Day 9 (done):** `day09_train_toy.py` trained the toy SAE. recon 0.49→0.0048, L0→~3, **8/8 true features recovered @ >0.99 cosine.** Milestone: the SAE method works, validated on known ground truth. Next: Day 10 scale to real GPT-2 activations (Colab).
- **Note (2026):** Per user preference, commits no longer include the Claude co-author trailer.
