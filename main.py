import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken  # pip install tiktoken
import numpy as np
import psutil
from colorama import init, Fore, Style

# init colorama
init(autoreset=True)


# ======================================================
# Helpers
# ======================================================
def memory_fraction():
    mem = psutil.virtual_memory()
    return mem.percent


def memory_usage():
    mem = psutil.virtual_memory()
    return mem.used


def simplifyNum(num, type="bytes"):
    num = int(num)

    if type == "bytes":
        if num > 1e15:
            return f"{num / 1e15:.2f}PB"
        elif num > 1e12:
            return f"{num / 1e12:.2f}TB"
        elif num > 1e9:
            return f"{num / 1e9:.2f}GB"
        elif num > 1e6:
            return f"{num / 1e6:.2f}MB"
        elif num > 1e3:
            return f"{num / 1e3:.2f}KB"
        else:
            return f"{num}B"

    elif type == "number":
        if num > 1e15:
            return f"{num / 1e15:.2f} quadrillion"
        elif num > 1e12:
            return f"{num / 1e12:.2f} trillion"
        elif num > 1e9:
            return f"{num / 1e9:.2f} billion"
        elif num > 1e6:
            return f"{num / 1e6:.2f} million"
        elif num > 1e3:
            return f"{num / 1e3:.2f} thousand"
        else:
            return f"{num:.2f}"
    else:
        raise ValueError("Invalid type")


# colored loggers
def log_info(msg):
    print(Fore.CYAN + msg + Style.RESET_ALL)


def log_success(msg):
    print(Fore.GREEN + msg + Style.RESET_ALL)


def log_warn(msg):
    print(Fore.YELLOW + msg + Style.RESET_ALL)


def log_error(msg):
    print(Fore.RED + msg + Style.RESET_ALL)


def log_progress(msg):
    print(Fore.WHITE + msg + Style.RESET_ALL)


# =====================================================
# Pre-start safety checks
# =====================================================

# Check if the system has at least 10GB of RAM available
if psutil.virtual_memory().available < 10 * 1000 * 1000 * 1000:
    log_warn("Less than 10GB of RAM available")
    log_error("Exiting...")
    exit(1)

# Check if the CPU has at least 10 cores
if psutil.cpu_count(logical=False) < 10:
    log_warn("Less than 10 CPU cores available")
    choice = input("Are you sure you want to continue? (y/n): ")
    if choice.lower() != "y":
        log_error("Exiting...")
        exit(1)
    else:
        cpuSufficient = False
else:
    cpuSufficient = True

# Check for a GPU
if not torch.cuda.is_available():
    log_warn("No GPU available")
    choice = input("Are you sure you want to run this on CPU only? (y/n): ")
    if choice.lower() != "y":
        log_error("Exiting...")
        exit(1)
    else:
        gpuSufficient = False
else:
    gpuSufficient = True

if not cpuSufficient:
    log_warn("[WARN]: RUNNING WITH LESS THAN 10 CPU CORES")
if not gpuSufficient:
    log_warn("[WARN]: RUNNING WITHOUT GPU")


# ======================================================
# 1. Tokenization
# ======================================================
data_file = "data.txt"
token_file = "tokens.npy"
enc = tiktoken.get_encoding("gpt2")

if not os.path.exists(token_file):
    log_info("Encoding data.txt into tokens (streaming)...")
    tokens_list = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens_list.extend(enc.encode(line + "\n"))
            if len(tokens_list) % 10000 == 0:
                log_progress(
                    f"Processed {simplifyNum(len(tokens_list), 'number')} tokens "
                    f"({Fore.MAGENTA}{simplifyNum(memory_usage(), 'bytes')}{Style.RESET_ALL} | "
                    f"{Fore.YELLOW}{memory_fraction():.1f}% RAM{Style.RESET_ALL})"
                )
            if memory_fraction() > 80:
                log_warn("Memory usage exceeded 80%")
                log_error("Exiting...")
                exit(1)
    np.save(token_file, np.array(tokens_list, dtype=np.int32))
    log_success(f"Saved {len(tokens_list)} tokens to {token_file}")
else:
    log_success("Found existing token file")


# ======================================================
# 2. Load tokens / GPT config
# ======================================================
tokens = np.load(token_file, mmap_mode="r")
vocab_size = enc.n_vocab


class GPTConfig:
    vocab_size = vocab_size
    n_layer = 4
    n_head = 4
    n_embd = 128
    block_size = min(128, max(8, len(tokens) // 100)) if len(tokens) > 0 else 128


log_info(
    f"Token dataset length: {Fore.MAGENTA}{len(tokens)}{Style.RESET_ALL} tokens | vocab size: {vocab_size}"
)
log_info(f"Using block_size = {Fore.BLUE}{GPTConfig.block_size}{Style.RESET_ALL}")

chunk_size = 10_000_000
num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
log_info(
    f"Will process dataset in {Fore.BLUE}{num_chunks}{Style.RESET_ALL} chunks (chunk_size={chunk_size})"
)


# ======================================================
# 3. GPT Model
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
# 4. Training or CLI
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
    total_steps = 2000
    log_interval = 100
    steps_done = 0
    start_time = time.time()

    log_info(
        f"Training new model for {Fore.BLUE}{total_steps}{Style.RESET_ALL} steps across {num_chunks} chunks..."
    )

    steps_per_chunk = max(1, total_steps // max(1, num_chunks))

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(len(tokens), start + chunk_size)
        data_chunk = torch.from_numpy(tokens[start:end].astype(np.int64))
        n = int(0.9 * len(data_chunk))
        train_data, val_data = data_chunk[:n], data_chunk[n:]

        log_info(
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

            if steps_done % log_interval == 0 or steps_done == 1:
                elapsed = time.time() - start_time
                recent = loss_history[-log_interval:]
                avg_loss = sum(recent) / len(recent)
                percent = 100 * steps_done / total_steps
                steps_left = max(0, total_steps - steps_done)
                est_time_left = (elapsed / max(1, steps_done)) * steps_left

                msg = (
                    f"[{steps_done}/{total_steps} | {Fore.BLUE}{percent:.1f}%{Style.RESET_ALL}] "
                    f"Loss: {Fore.CYAN}{loss.item():.4f}{Style.RESET_ALL} "
                    f"(avg {avg_loss:.4f}) "
                    f"Elapsed: {elapsed / 60:.1f}m ETA: {est_time_left / 60:.1f}m"
                )
                if device == "cuda":
                    mem = torch.cuda.memory_allocated() / (1024**2)
                    msg += f" | GPU Mem: {Fore.MAGENTA}{mem:.1f} MB{Style.RESET_ALL}"
                msg += (
                    f" | RAM%: {Fore.YELLOW}{memory_fraction():.1f}%{Style.RESET_ALL}"
                )
                log_progress(msg)

            if memory_fraction() > 80.0:
                log_warn("⚠️ Memory above 80% — checkpointing...")
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    model_file,
                )
                try:
                    del train_data, val_data, data_chunk
                except Exception:
                    pass
                if device == "cuda":
                    torch.cuda.empty_cache()
                break

        log_info(f"Saving checkpoint after chunk {chunk_idx + 1} ...")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            model_file,
        )
        try:
            del train_data, val_data, data_chunk
        except Exception:
            pass
        if device == "cuda":
            torch.cuda.empty_cache()

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
    log_success("Training complete. Final checkpoint and metadata saved.")

else:
    log_success("Found existing model. Loading...")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    log_success("Model loaded. Entering CLI mode.")

    while True:
        try:
            prompt = input(Fore.CYAN + "\nYou> " + Style.RESET_ALL)
            if not prompt.strip():
                continue
            if prompt.lower() in {"quit", "exit"}:
                log_info("Exiting CLI.")
                break
            start = torch.tensor([enc.encode(prompt)], device=device)
            out = model.generate(start, max_new_tokens=100)
            decoded = enc.decode(out[0].tolist())
            print(Fore.GREEN + "AI> " + Style.RESET_ALL + decoded)
        except KeyboardInterrupt:
            log_info("\nExiting CLI.")
            break
