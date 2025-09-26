import os
import json
import time
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken  # pip install tiktoken
import numpy as np
import psutil


def memory_fraction():
    mem = psutil.virtual_memory()
    return mem.percent  # percentage of RAM used


# ======================================================
# 1. Download pre-extracted Wikipedia text
# ======================================================
data_url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.txt"
data_file = "data.txt"

#
# Load token file lazily and set vocab / chunk parameters
# ------------------------------------------------------
tokens = np.load(token_file, mmap_mode="r")
vocab_size = enc.n_vocab
GPTConfig.vocab_size = vocab_size

# tune block_size based on dataset length if you want it smaller
# keep it reasonable; if dataset is tiny, reduce block_size accordingly
data_len = len(tokens)
GPTConfig.block_size = min(128, max(8, data_len // 100)) if data_len > 0 else 128
print(f"Token dataset length: {data_len} tokens | vocab size: {vocab_size}")
print(f"Using block_size = {GPTConfig.block_size}")

# chunking parameters (adjust chunk_size to fit your RAM constraints)
chunk_size = 10_000_000  # tokens per chunk (tweak smaller if still OOM)
num_chunks = (data_len + chunk_size - 1) // chunk_size
print(f"Will process dataset in {num_chunks} chunks (chunk_size={chunk_size})")

# ======================================================
# 2–3. Stream tokenization and save to disk
# ======================================================
enc = tiktoken.get_encoding("gpt2")
token_file = "tokens.npy"

if not os.path.exists(token_file):
    print("Encoding data.txt into tokens (streaming)...")
    tokens = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens.extend(enc.encode(line + "\n"))
            if len(tokens) % 10000 == 0:
                print(f"Processed {len(tokens)} tokens")
    np.save(token_file, np.array(tokens, dtype=np.int32))
    print(f"Saved {len(tokens)} tokens to {token_file}")
else:
    print("Found existing token file")


# ======================================================
# 4. GPT config (set after tokens loaded)
# ======================================================
class GPTConfig:
    # placeholders — real values set below after token file is loaded
    vocab_size = None
    n_layer = 4
    n_head = 4
    n_embd = 128
    block_size = 128  # default; overridden later if needed


# Note: we will fill GPTConfig.vocab_size and GPTConfig.block_size
# after we load the token file below.


# ======================================================
# 5. GPT model
# ======================================================
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=config.n_embd,
                    nhead=config.n_head,
                    dim_feedforward=4 * config.n_embd,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x, x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens=200):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -GPTConfig.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


# ======================================================
# 6. Training or Load Model
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(GPTConfig).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

loss_history = []

model_file = "gpt_model.pt"
meta_file = "training_metadata.json"


def get_batch(split, batch_size=32):
    data_split = train_data if split == "train" else val_data
    if len(data_split) <= GPTConfig.block_size:
        raise RuntimeError("Dataset too small for block_size")
    ix = torch.randint(len(data_split) - GPTConfig.block_size, (batch_size,))
    x = torch.stack([data_split[i : i + GPTConfig.block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + 1 + GPTConfig.block_size] for i in ix])
    return x.to(device), y.to(device)


if not os.path.exists(model_file):
    # total training steps you want overall; you can scale this
    total_steps = 2000
    log_interval = 100
    steps_done = 0
    start_time = time.time()

    print(f"Training new model for {total_steps} steps across {num_chunks} chunks...")

    # how many steps per chunk (simple division — tweak if necessary)
    steps_per_chunk = max(1, total_steps // max(1, num_chunks))

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(len(tokens), start + chunk_size)
        # create torch tensor view of this chunk (does not allocate whole dataset)
        data_chunk = torch.from_numpy(tokens[start:end].astype(np.int64))
        n = int(0.9 * len(data_chunk))
        train_data, val_data = data_chunk[:n], data_chunk[n:]

        print(
            f"=== Chunk {chunk_idx + 1}/{num_chunks} | tokens {start}:{end} ({len(data_chunk)}) ==="
        )

        for step in range(1, steps_per_chunk + 1):
            xb, yb = get_batch("train")
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            steps_done += 1

            # Logging
            if steps_done % log_interval == 0 or steps_done == 1:
                elapsed = time.time() - start_time
                recent = loss_history[-log_interval:]
                avg_loss = sum(recent) / len(recent)
                percent = 100 * steps_done / total_steps
                steps_left = max(0, total_steps - steps_done)
                est_time_left = (elapsed / max(1, steps_done)) * steps_left
                msg = (
                    f"[{steps_done}/{total_steps} | {percent:.1f}%] "
                    f"Loss: {loss.item():.4f} (avg {avg_loss:.4f}) "
                    f"Elapsed: {elapsed / 60:.1f}m ETA: {est_time_left / 60:.1f}m"
                )
                if device == "cuda":
                    mem = torch.cuda.memory_allocated() / (1024**2)
                    msg += f" | GPU Mem: {mem:.1f} MB"
                msg += f" | RAM%: {memory_fraction():.1f}%"
                print(msg)

            # Memory guard: if system RAM usage climbs above threshold, checkpoint & free
            if memory_fraction() > 80.0:
                print(
                    "⚠️ Memory above 80% — checkpointing and clearing chunk to avoid OOM."
                )
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    model_file,
                )
                # free chunk tensors & clear caches
                try:
                    del train_data, val_data, data_chunk
                except Exception:
                    pass
                if device == "cuda":
                    torch.cuda.empty_cache()
                # break out of steps for this chunk (will continue with next chunk)
                break

        # end per-chunk loop — save checkpoint after chunk completes or early exit
        print(f"Saving checkpoint after chunk {chunk_idx + 1} ...")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            model_file,
        )
        # explicitly free the chunk if still present
        try:
            del train_data, val_data, data_chunk
        except Exception:
            pass
        if device == "cuda":
            torch.cuda.empty_cache()

    # After all chunks, write metadata
    metadata = {
        "config": {
            "vocab_size": GPTConfig.vocab_size,
            "n_layer": GPTConfig.n_layer,
            "n_head": GPTConfig.n_head,
            "n_embd": GPTConfig.n_embd,
            "block_size": GPTConfig.block_size,
        },
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else None,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print("Training complete. Final checkpoint and metadata saved.")

else:
    print("Found existing model. Loading...")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print("Model loaded. Entering CLI mode.")

    # CLI loop
    while True:
        try:
            prompt = input("\nYou> ")
            if not prompt.strip():
                continue
            if prompt.lower() in {"quit", "exit"}:
                print("Exiting CLI.")
                break
            start = torch.tensor([enc.encode(prompt)], device=device)
            out = model.generate(start, max_new_tokens=100)
            decoded = enc.decode(out[0].tolist())
            print("AI>", decoded)
        except KeyboardInterrupt:
            print("\nExiting CLI.")
            break
