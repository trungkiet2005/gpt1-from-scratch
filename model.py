import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import time
import os

# Step 1: Define hyperparameters
batch_size = 32        # Reduced for P100 memory
block_size = 256       # Smaller context to reduce load
max_iters = 30000      # Training iterations
eval_interval = 500    # Evaluate every 500 steps
learning_rate = 3e-4   # Learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200        # Fewer eval iterations to start faster
n_embd = 512           # Embedding dimension
n_head = 12             # Attention heads
n_layer = 12            # Transformer layers
dropout = 0.1          # Dropout

# Step 2: Load and process data in chunks
def read_in_chunks(file_path, chunk_size=1024*1024):
    with open(file_path, 'r', encoding='utf-8') as f:
        while chunk := f.read(chunk_size):
            yield chunk

print("Step 1: Loading data...")
start_time = time.time()
text = ''.join(read_in_chunks('dataset.txt')).lower()  # Convert to lowercase
print(f"Data loaded in {time.time() - start_time:.2f} seconds")

# Step 3: Create vocabulary and encode/decode functions
print("Step 2: Building vocabulary...")
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s.lower()]  # Convert to lowercase before encoding
decode = lambda l: ''.join([itos[i] for i in l])
print(f"Vocabulary size: {vocab_size}")

# Step 4: Load or encode data
encoded_data_path = 'encoded_data.pt'
if os.path.exists(encoded_data_path):
    print("Step 3: Loading pre-encoded data...")
    data = torch.load(encoded_data_path, map_location='cpu')
    print(f"Loaded encoded data from '{encoded_data_path}'")
else:
    print("Step 3: Encoding data...")
    data = torch.tensor(encode(text), dtype=torch.long)
    torch.save(data, encoded_data_path)
    print(f"Encoded data saved to '{encoded_data_path}'")

# Split data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

# Step 5: Data loading function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # print(f"  get_batch: Selected {split} data, size: {len(data)}")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(f"  get_batch: Indices generated, shape: {ix.shape}")
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    # print(f"  get_batch: Tensors stacked, x shape: {x.shape}, y shape: {y.shape}")
    x, y = x.to(device), y.to(device)
    # print(f"  get_batch: Tensors moved to {device}")
    return x, y

# Step 6: Loss estimation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # print(f"  Evaluating {split}, iter {k}/{eval_iters}")
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Step 7: Self-attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# Step 8: Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Step 9: Feed-forward network
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Step 10: Transformer block
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Step 11: GPT language model
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Step 12: Initialize model and optimizer
print("Step 4: Initializing model...")
model = GPTLanguageModel()
m = model.to(device)
print(f"Model parameters: {sum(p.numel() for p in m.parameters()) / 1e6} M")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Step 13: Check GPU memory
print("Step 5: Checking GPU memory...")
if device == 'cuda':
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

# # Step 14: Training loop
# print("Step 6: Starting training...")
# start_time = time.time()
# for iter in tqdm(range(max_iters), desc="Training"):
#     if iter % eval_interval == 0 or iter == max_iters - 1:
#         # print("  Starting loss estimation...")
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
#     step_start = time.time()
#     xb, yb = get_batch('train')
#     # print("  Batch loaded")
    
#     logits, loss = model(xb, yb)
#     # print("  Forward pass done")
    
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     # print("  Backward pass done")
    
#     optimizer.step()
    


# final_model_path = os.path.join('/kaggle/working', 'final_model.pt')
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
# }, final_model_path)
# print(f"Step 6: Final model saved to {final_model_path}")

# Step 15: Generate sample text
# print("Step 7: Generating sample...")
# context = torch.tensor([encode("start of a poem")], dtype=torch.long, device=device)
# generated = m.generate(context, max_new_tokens=500)[0].tolist()
# print(decode(generated))
