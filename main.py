import os
import json
import time
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken  # pip install tiktoken
import numpy as np

# ======================================================
# 1. Download pre-extracted Wikipedia text
# ======================================================
data_url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.txt"
data_file = "data.txt"

if not os.path.exists(data_file):
    print("Downloading pre-extracted Wikipedia text...")
    r = requests.get(data_url, stream=True)
    with open(data_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete:", data_file)
else:
    print("Found existing data.txt")

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
# 4. GPT config
# ======================================================
class GPTConfig:
    vocab_size = vocab_size
    n_layer = 4
    n_head = 4
    n_embd = 128
    block_size = min(128, max(8, len(train_data) // 100))


print(f"Using block_size = {GPTConfig.block_size}")


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
    total_steps = 2000  # increase when training with big data
    log_interval = 100

    print(f"Training new model for {total_steps} steps...")
    start_time = time.time()

    for step in range(1, total_steps + 1):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        # Logging
        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            avg_loss = sum(loss_history[-log_interval:]) / min(
                len(loss_history), log_interval
            )
            percent = 100 * step / total_steps
            steps_left = total_steps - step
            est_time_left = (elapsed / step) * steps_left
            msg = (
                f"[{step}/{total_steps} | {percent:.1f}%] "
                f"Loss: {loss.item():.4f} (avg {avg_loss:.4f}) "
                f"Elapsed: {elapsed / 60:.1f}m ETA: {est_time_left / 60:.1f}m"
            )
            if device == "cuda":
                mem = torch.cuda.memory_allocated() / (1024**2)
                msg += f" | GPU Mem: {mem:.1f} MB"
            print(msg)

    # Save model + metadata
    print("Saving model and metadata...")
    torch.save(model.state_dict(), model_file)

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

    print(
        "Training complete. Model saved to gpt_model.pt, metadata saved to training_metadata.json"
    )

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
