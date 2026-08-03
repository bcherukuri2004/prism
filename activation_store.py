"""
Day 5 — Store the smoothies properly (so training can scale).

Problem: Day 4 loaded all 32,000 smoothies into RAM at once. That's fine now,
but later (in Colab) we'll want MILLIONS, which won't fit in memory.

Solution: an "activation store" that
  1. splits the pile into several smaller files on disk ("shards"), and
  2. STREAMS batches: keep only a chunk in memory, shuffle it, hand out small
     batches, and quietly pull in the next shard when the chunk runs low.

This "shuffle buffer" idea is how real data pipelines feed models without ever
holding the whole dataset in memory. This file is reusable — later days import
ActivationStore from here.
"""

import os
import glob
import torch


def build_shards(src_path="activations/layer6_resid.pt",
                 shard_dir="activations/shards",
                 shard_size=8000):
    """Cut one big saved activation tensor into several shard files.

    (In a real large run you'd write shards directly while collecting. Here we
    just slice the Day-4 file so we have multiple shards to practice streaming.)
    """
    os.makedirs(shard_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    if existing:
        print(f"Shards already exist ({len(existing)} files) — skipping build.")
        return existing

    activations = torch.load(src_path)["activations"]          # [N, 768]
    paths = []
    for i, start in enumerate(range(0, activations.shape[0], shard_size)):
        shard = activations[start:start + shard_size].clone()
        path = os.path.join(shard_dir, f"shard_{i:03d}.pt")
        torch.save(shard, path)
        paths.append(path)
    print(f"Wrote {len(paths)} shards (up to {shard_size} rows each) to {shard_dir}/")
    return paths


class ActivationStore:
    """Streams activation batches from disk shards, never loading everything at once.

    It keeps a `buffer` of rows in memory. When the buffer runs low it loads the
    next shard, appends it, and reshuffles. Batches are served off the front.
    """

    def __init__(self, shard_dir="activations/shards", batch_size=64,
                 buffer_size=16000, device="cpu", shuffle=True):
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards in {shard_dir!r} — run build_shards() first.")
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.shuffle = shuffle

    def batches(self):
        """Yield batches of shape [batch_size, 768] until the data runs out."""
        buffer = torch.empty(0)      # rows waiting to be served
        next_shard = 0

        while True:
            # 1) refill the buffer from shards until it's full (or shards run out)
            while buffer.shape[0] < self.buffer_size and next_shard < len(self.shard_paths):
                shard = torch.load(self.shard_paths[next_shard])
                buffer = shard if buffer.numel() == 0 else torch.cat([buffer, shard], dim=0)
                next_shard += 1
                if self.shuffle:
                    buffer = buffer[torch.randperm(buffer.shape[0])]

            # 2) if we can't fill even one batch and no shards remain, we're done
            if buffer.shape[0] < self.batch_size:
                break

            # 3) serve one batch off the front, keep the rest for next time
            batch, buffer = buffer[:self.batch_size], buffer[self.batch_size:]
            yield batch.to(self.device)


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Step 1: make shards out of the Day-4 pile.
    build_shards()

    # Step 2: stream batches from them.
    store = ActivationStore(batch_size=64, buffer_size=16000, device=device)
    print(f"Streaming from {len(store.shard_paths)} shards on {device}...\n")

    total = 0
    for i, batch in enumerate(store.batches()):
        if i < 3:
            # peek at a few batches (mean ~0 is expected: the residual stream is roughly centered)
            print(f"  batch {i}: shape {tuple(batch.shape)}, "
                  f"mean {batch.mean():.3f}, device {batch.device.type}")
        total += 1

    print(f"\nServed {total} batches of 64 = ~{total * 64:,} rows "
          f"(from the 32,000 we collected), never holding more than "
          f"{store.buffer_size:,} in memory at once.")
