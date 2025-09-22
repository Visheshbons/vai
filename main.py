import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import csv
import math
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# ===== Tokenization & Vocab =====
def tokenize(text):
    return text.strip().split()

class Vocab:
    def __init__(self):
        self.word2idx = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.idx2word = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.idx = 4

    def add_sentence(self, sentence):
        for word in tokenize(sentence):
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx +=1

    def encode(self, sentence, add_sos_eos=True):
        tokens = tokenize(sentence)
        ids = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens]
        if add_sos_eos:
            ids = [self.word2idx["<SOS>"]] + ids + [self.word2idx["<EOS>"]]
        return ids

    def decode(self, ids):
        words = [self.idx2word.get(i, "<UNK>") for i in ids]
        # strip SOS/EOS
        if words and words[0] == "<SOS>": words = words[1:]
        if words and words[-1] == "<EOS>": words = words[:-1]
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

# ===== Dataset =====
class ChatDataset(Dataset):
    def __init__(self, csv_file, vocab):
        self.data = []
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append((row['input'], row['response']))
                vocab.add_sentence(row['input'])
                vocab.add_sentence(row['response'])
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp, out = self.data[idx]
        return torch.tensor(self.vocab.encode(inp)), torch.tensor(self.vocab.encode(out))

def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_in = max(len(s) for s in inputs)
    max_out = max(len(s) for s in targets)
    padded_in = torch.zeros(len(inputs), max_in, dtype=torch.long)
    padded_out = torch.zeros(len(targets), max_out, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_in[i, :len(inp)] = inp
        padded_out[i, :len(tgt)] = tgt
    return padded_in, padded_out

# ===== Positional Encoding =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ===== Transformer Seq2Seq =====
class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_mask, tgt_mask = None, self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_embed = self.positional_encoding(self.embedding(src))
        tgt_embed = self.positional_encoding(self.embedding(tgt))
        out = self.transformer(src_embed, tgt_embed, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.fc_out(out)

# Auto-adjust hyperparameters based on dataset size
def auto_hparams(dataset_size):
    if dataset_size < 50:
        epochs = 60
        temperature = 0.3
    elif dataset_size < 200:
        epochs = 40
        temperature = 0.7
    elif dataset_size < 1000:
        epochs = 20
        temperature = 0.9
    else:
        epochs = 10
        temperature = 1.0
    return epochs, temperature

# ===== Training =====
def train(model, dataloader, vocab, epochs=10, lr=0.001, device="cpu"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])

    print(f"{Fore.CYAN}Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])  # predict next tokens
            loss = criterion(output.reshape(-1, len(vocab)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss/len(dataloader)
        print(f"{Fore.GREEN}Epoch {epoch+1}/{epochs} - Loss: {Fore.YELLOW}{avg_loss:.4f}")
    print(f"{Fore.GREEN}Training completed!")

# ===== Inference =====
def generate_response(model, vocab, sentence, max_len=20, temperature=1.0, device="cpu"):
    model.eval()
    src = torch.tensor(vocab.encode(sentence)).unsqueeze(0).to(device)
    src_embed = model.positional_encoding(model.embedding(src))
    memory = model.transformer.encoder(src_embed)

    tgt = torch.tensor([[vocab.word2idx["<SOS>"]]]).to(device)
    output_words = []

    for _ in range(max_len):
        tgt_embed = model.positional_encoding(model.embedding(tgt))
        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        out = model.transformer.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        logits = model.fc_out(out[:, -1])
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        if next_token == vocab.word2idx["<EOS>"]:
            break
        output_words.append(next_token)
        tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)

    return vocab.decode(output_words)

# ===== Main =====
if __name__ == "__main__":
    print(f"{Fore.MAGENTA}VAI Chatbot Starting...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.CYAN}Using device: {device}")

    vocab = Vocab()
    dataset = ChatDataset("data.csv", vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Auto hyperparams
    epochs, temperature = auto_hparams(len(dataset))
    print(f"{Fore.BLUE}Dataset size: {len(dataset)} → Training for {epochs} epochs, temperature={temperature}")

    model = TransformerChatbot(len(vocab))
    train(model, dataloader, vocab, epochs=epochs, device=device)

    print(f"\n{Fore.CYAN}Chat started! Type 'quit' to exit.")
    while True:
        user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}")
        if user_input.lower() in ["quit", "exit"]:
            print(f"{Fore.GREEN}Goodbye!")
            break

        response = generate_response(model, vocab, user_input.lower(), device=device, temperature=temperature)
        print(f"{Fore.RED}Bot: {Style.RESET_ALL}{response}")
